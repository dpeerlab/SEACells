from tqdm import tqdm
import pyranges as pr
from sklearn.metrics import pairwise_distances
from scipy.stats import rankdata

import numpy as np
import pandas as pd
import scanpy as sc

def compute_ArchR_vanilla():
    """
    @Manu: This is loaded from a saved file in the notebook. Should we add it in the package?
    """
    raise NotImplementedError

def prepare_multiome_anndata(atac_ad, rna_ad, SEACell_label='SEACell'):
    """
    todo: Documentation
    @Manu: rna_ad.X, atac_ad.X must be raw counts??

    """
    from scipy.sparse import csr_matrix
    
    atac_ad.var['GC_bin'] = np.digitize(atac_ad.var['GC'], np.linspace(0, 1, 50))
    atac_ad.var['log_n_counts'] = np.ravel(np.log10(atac_ad.X.sum(axis=0)))
    atac_ad.var['counts_bin'] = np.digitize(atac_ad.var['log_n_counts'],
                                            np.linspace(atac_ad.var['log_n_counts'].min(),
                                                        atac_ad.var['log_n_counts'].max(), 50))

    # Perform inverse document frequency transformation on count matrix
    from sklearn.feature_extraction.text import TfidfTransformer
    mat = atac_ad.X.astype(int)
    tfidf = TfidfTransformer().fit(mat)
    atac_ad.layers['TFIDF'] = tfidf.transform(mat)

    metacells = atac_ad.obs[SEACell_label].astype(str).unique()
    metacells = metacells[atac_ad.obs[SEACell_label].value_counts()[metacells] > 1]

    # Summary matrix
    summ_matrix = pd.DataFrame(0.0, index=metacells, columns=atac_ad.var_names)

    for m in tqdm(summ_matrix.index):
        cells = atac_ad.obs_names[atac_ad.obs[SEACell_label] == m]
        summ_matrix.loc[m, :] = np.ravel(atac_ad[cells, :].layers['TFIDF'].sum(axis=0))

    # Construct anndata from ATAC SEACells post-aggregation
    atac_meta_ad = sc.AnnData(summ_matrix)
    atac_meta_ad.X = csr_matrix(atac_meta_ad.X)
    atac_meta_ad.obs_names, atac_meta_ad.var_names = summ_matrix.index.astype(str), summ_matrix.columns

    # Now summarize RNA anndata according to ATAC SEACell
    metacells = rna_ad.obs[SEACell_label].astype(str).unique()
    metacells = metacells[rna_ad.obs[SEACell_label].value_counts()[metacells] > 1]

    # TODO: Check if Summary matrix constructed from raw counts
    summ_matrix = pd.DataFrame(0.0, index=metacells, columns=rna_ad.var_names)

    for m in tqdm(summ_matrix.index):
        cells = rna_ad.obs_names[atac_ad.obs[SEACell_label] == m]
        summ_matrix.loc[m, :] = np.ravel(rna_ad[cells, :].X.sum(axis=0))

    # Ann data
    rna_meta_ad = sc.AnnData(summ_matrix)
    rna_meta_ad.X = csr_matrix(rna_meta_ad.X)
    rna_meta_ad.obs_names, rna_meta_ad.var_names = summ_matrix.index.astype(str), rna_meta_ad.var_names

    rna_meta_ad.var['highly_variable'] = rna_ad.var['highly_variable']

    return atac_ad, atac_meta_ad, rna_meta_ad

def pyranges_from_strings(pos_list):
    """
    TODO: Documentation
    """
    # Chromosome and positions
    chr = pos_list.str.split(':').str.get(0)
    start = pd.Series(pos_list.str.split(':').str.get(1)).str.split('-').str.get(0)
    end = pd.Series(pos_list.str.split(':').str.get(1)).str.split('-').str.get(1)

    # Create ranges
    gr = pr.PyRanges(chromosomes=chr, starts=start, ends=end)
    return gr


def pyranges_to_strings(peaks):
    """
    TODO: Documentation
    """
    # Chromosome and positions
    chr = peaks.Chromosome.astype(str).values
    start = peaks.Start.astype(str).values
    end = peaks.End.astype(str).values

    # Create ranges
    gr = chr + ':' + start + '-' + end

    return gr

def load_transcripts(path_to_hg38gtf):
    gtf = pr.read_gtf(path_to_hg38gtf)
    gtf.Chromosome = 'chr' + gtf.Chromosome.astype(str)
    transcripts = gtf[gtf.Feature == 'transcript']
    return transcripts

def dorc_func(gene,
              atac_exprs,
              rna_exprs,
              atac_ad,
              peaks_pr,
              transcripts,
              span):
    # Longest transcripts
    gene_transcripts = transcripts[transcripts.gene_name == gene]
    if len(gene_transcripts) == 0:
        return 0
    longest_transcript = gene_transcripts[
        np.arange(len(gene_transcripts)) == np.argmax(gene_transcripts.End - gene_transcripts.Start)]
    start = longest_transcript.Start.values[0] - span
    end = longest_transcript.End.values[0] + span

    # Gene span
    gene_pr = pr.from_dict({'Chromosome': [longest_transcript.Chromosome.values[0]],
                            'Start': [start],
                            'End': [end]})
    gene_peaks = peaks_pr.overlap(gene_pr)
    if len(gene_peaks) == 0:
        return 0
    gene_peaks_str = pyranges_to_strings(gene_peaks)

    # correlations
    if type(atac_exprs) is sc.AnnData:
        X = pd.DataFrame(atac_exprs[:, gene_peaks_str].X.todense().T)
    else:
        X = atac_exprs.loc[:, gene_peaks_str].T

    cors = 1 - np.ravel(pairwise_distances(np.apply_along_axis(rankdata, 1, X.values),
                                           rankdata(rna_exprs[gene].T.values).reshape(1, -1),
                                           metric='correlation'))
    cors = pd.Series(cors, index=gene_peaks_str)
    cors = cors[cors > 0.01]

    # Random background
    df = pd.DataFrame(1.0, index=cors.index, columns=['cor', 'pval'])
    df['cor'] = cors
    for p in df.index:
        try:
            rand_peaks = np.random.choice(atac_ad.var_names[(atac_ad.var['GC_bin'] == atac_ad.var['GC_bin'][p]) & \
                                                            (atac_ad.var['counts_bin'] == atac_ad.var['counts_bin'][
                                                                p])], 100, False)
        except:
            rand_peaks = np.random.choice(atac_ad.var_names[(atac_ad.var['GC_bin'] == atac_ad.var['GC_bin'][p]) & \
                                                            (atac_ad.var['counts_bin'] == atac_ad.var['counts_bin'][
                                                                p])], 100, True)

        if type(atac_exprs) is sc.AnnData:
            X = pd.DataFrame(atac_exprs[:, rand_peaks].X.todense().T)
        else:
            X = atac_exprs.loc[:, rand_peaks].T

        rand_cors = 1 - np.ravel(pairwise_distances(np.apply_along_axis(rankdata, 1, X.values),
                                                    rankdata(rna_exprs[gene].T.values).reshape(1, -1),
                                                    metric='correlation'))

        m = np.mean(rand_cors)
        v = np.std(rand_cors)

        from scipy.stats import norm
        df.loc[p, 'pval'] = 1 - norm.cdf(cors[p], m, v)

    return df

def get_gene_peak_associations(atac_aggregated_ad,
                               rna_aggregated_ad,
                               atac_ad,
                               transcripts,
                               span=100000):
    """

    """
    atac_exprs = pd.DataFrame(atac_aggregated_ad.X.todense(),
                                   index=atac_aggregated_ad.obs_names, columns=atac_aggregated_ad.var_names)
    rna_exprs = pd.DataFrame(rna_aggregated_ad.X.todense(),
                             index=rna_aggregated_ad.obs_names, columns=rna_aggregated_ad.var_names)

    peaks_pr = pyranges_from_strings(atac_aggregated_ad.var_names)

    from joblib import Parallel, delayed
    use_genes = rna_aggregated_ad.var_names
    gene_res = Parallel(n_jobs=1)(delayed(dorc_func)(gene,
                                                     atac_exprs,
                                                     rna_exprs,
                                                     atac_ad,
                                                     peaks_pr,
                                                     transcripts,
                                                     span)
                                  for gene in tqdm(use_genes))
    gene_res = pd.Series(gene_res, index=use_genes)
    return gene_res

def get_gene_scores(atac_aggregated_ad, gene_peak_associations, pval_cutoff = 1e-1):
    """
    Gene scores are computed as the aggregate accessibility of all peaks associated with a gene.
    See .get_gene_peak_associations() for details on how gene-peak associations are computed.
    """
    gene_scores = pd.DataFrame(0.0, index=atac_aggregated_ad.obs_names, columns=gene_peak_associations.index)

    for gene in tqdm(gene_scores.columns):
        df = gene_peak_associations[gene]
        if type(df) is int:
            continue
        gene_peaks = df.index[df['pval'] < pval_cutoff]
        gene_scores[gene] = np.ravel(np.dot(atac_aggregated_ad[:, gene_peaks].X.todense(),
                                            df.loc[gene_peaks, 'cor']))
    gene_scores = gene_scores.loc[:, (gene_scores.sum() >= 0)]
    return gene_scores
