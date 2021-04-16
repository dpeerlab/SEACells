import scanpy as sc
import anndata
import palantir
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)

def get_NLLs(ad, 
             key = 'X_pca',
             n_neighbours = 5,
             TRAIN_SPLIT = 1,
             cluster = 'celltype', 
             verbose = True):
    """
    Compute the NLL of each cell under the Gaussian defined by cells belonging to (1) its own metacell
    and (2) the metacells of its nearest neighbours. The multivariate Gaussian is defined in the space specified by key
    and distances are computed in the diffusion component space.
    """

    label_df = ad.obs[['Metacell', cluster]]

    # Compute neighbours in diffusion component space
    if verbose:
        print(f'Computing {n_neighbours} neighbours in diffusion component space and Gaussian from {key} space.')
    components = pd.DataFrame(ad.obsm[key]).set_index(ad.obs_names)
    dm_res = palantir.utils.run_diffusion_maps(components)
    dc = palantir.utils.determine_multiscale_space(dm_res, n_eigs=10)

    cts = label_df.groupby('Metacell').count()
    if TRAIN_SPLIT < 1:
        if verbose:
            print(f'Clipping to metacells with at least {1/(1-TRAIN_SPLIT)} cells')
        sufficient = cts[cts[cluster]>1/(1-TRAIN_SPLIT)].index
    else:
        if verbose:
            print(f'Clipping to metacells with at least 3 cells.')
        # Require at least 3 cells per metacell
        sufficient = cts[cts[cluster]>=3].index
    
    clip_label_df = label_df[label_df['Metacell'].isin(sufficient)]
    
    if verbose:
        print(f'Dropping {len(label_df.Metacell.unique())-len(sufficient)} metacell(s) due to insufficient size.')

    # Subset to only metacells
    metacells_dcs = dc.merge(clip_label_df.Metacell, left_index=True, right_index=True).groupby('Metacell').mean()
    labeled_dcs = dc.merge(clip_label_df.Metacell, left_index=True, right_index=True).set_index('Metacell')

    label_df = ad.obs[['Metacell', cluster]]
    cts = label_df.groupby('Metacell').count()
    if TRAIN_SPLIT < 1:
        if verbose:
            print(f'Clipping to metacells with at least {1/(1-TRAIN_SPLIT)} cells')
        sufficient = cts[cts[cluster]>1/(1-TRAIN_SPLIT)].index
    else:
        if verbose:
            print(f'Clipping to metacells with at least 3 cells.')
        # Require at least 3 cells per metacell
        sufficient = cts[cts[cluster]>=3].index

    if verbose:
        print(f'Dropping {len(label_df.Metacell.unique())-len(sufficient)} metacell(s) due to insufficient size.')

    dropped = label_df[~label_df['Metacell'].isin(sufficient)]

    label_df = label_df[label_df['Metacell'].isin(sufficient)]
    label_df['Metacell'] = label_df['Metacell'].astype(str).astype('category')

    components = pd.DataFrame(ad.obsm[key]).set_index(ad.obs_names)
    components = components.loc[label_df.index]

    # Compute neighbours in diffusion component space
    if verbose:
        print(f'Computing {n_neighbours} neighbours in diffusion component space and Gaussian from {key} space.')
    dm_res = palantir.utils.run_diffusion_maps(components)
    dc = palantir.utils.determine_multiscale_space(dm_res, n_eigs=10)

    # Compute DC per metacell
    metacells_dcs = dc.join(label_df.Metacell, how='inner').groupby('Metacell').mean()

    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=n_neighbours)
    nbrs = neigh.fit(metacells_dcs)
    dists, nbrs = nbrs.kneighbors()
    dists = pd.DataFrame(dists).set_index(metacells_dcs.index)

    nbr_cells = np.array(metacells_dcs.index)[nbrs]

    metacells_nbrs = pd.DataFrame(nbr_cells)
    metacells_nbrs.index = metacells_dcs.index
    metacells_nbrs.columns += 1

    # Get cluster type of neighbors to ensure they match the metacell cluster
    metacells_with_cluster = metacells_dcs.join(label_df.groupby('Metacell').first(), how='inner')
    clusters_nbrs = pd.DataFrame(np.array(metacells_with_cluster[cluster].values)[nbrs])
    clusters_nbrs.index = metacells_with_cluster.index
    clusters_nbrs.columns += 1

    from scipy.spatial.distance import pdist, squareform

    mc_components_mean = components.merge(label_df['Metacell'], left_index=True, right_index=True).groupby('Metacell').mean()

    mc_dists = pd.DataFrame(squareform(pdist(mc_components_mean)))
    mc_dists.index = mc_components_mean.index
    mc_dists.columns = mc_components_mean.index

    dists = []
    for mc, row in metacells_nbrs.iterrows():
        dists.append(mc_dists[row].loc[mc].values)

    nbr_dists = pd.DataFrame(dists).set_index(metacells_nbrs.index)
    nbr_dists.columns += 1

    # Check if MC neighbour matches cluster type of metacell
    df = clusters_nbrs.join(ad.obs.groupby('Metacell').agg(lambda x:x.value_counts().index[0])[cluster])
    nbr_match = df.eq(df[cluster], axis=0).drop(cluster, axis=1)

    neighbours = pd.concat({'Metacell':metacells_nbrs, f'{cluster}':clusters_nbrs, 'Distance':nbr_dists, f'{cluster}_match':nbr_match}, axis=1)

    from scipy.stats import multivariate_normal as mv_g

    # Get PC components
    train_NLL_means = []
    train_NLL_stdevs = []
    test_NLL_means = []
    test_NLL_stdevs = []
    mcs = []
    n_singles = 0

    mc_params = {}
    mc_splits = {}

    for mc in label_df['Metacell'].dropna().unique():
        # Subset to cells assigned to this metacell
        subset = label_df[label_df['Metacell']==mc].index

        # Choose a train/test subset 
        ix = np.arange(len(subset))
        np.random.shuffle(ix)
        train = int(TRAIN_SPLIT*len(subset))
        train_ix = subset[ix[:train]]
        test_ix = subset[ix[train:]]

        components_subset = components.loc[train_ix]
        # Learn a multivariate gaussian from the train set

        cov = components_subset.cov().values
        mean = np.mean(components_subset.values, axis=0)

        mc_params[mc] = (mean,cov)
        mc_splits[mc] = (train_ix, test_ix)

    NLL_df = {'cell_id':[], 'self': [], 'train':[], 'mc_id': []}

    for col in range(n_neighbours):
        NLL_df[col+1] = []

    # Now iterate through all metacells/ neighbours and compute separation and 
    for index, row in metacells_nbrs.iterrows():
        # Subset to components for just this metacell
        train_ix, test_ix = mc_splits[index]
        if TRAIN_SPLIT < 1:
            splits = [(train_ix, True), (test_ix, False)]
        else:
            splits = [(train_ix, True)]

        # Get individual logliks:
        for ix, truth in splits:
            components_subset = components.loc[ix]

            NLL_df['mc_id'] += [index]*len(components_subset)
            NLL_df['cell_id'] += list(components_subset.index)
            NLL_df['train'] += [truth]*len(components_subset)
            mean, cov = mc_params[index]

            # Compute the probability under the assigned cell
            gaussian = mv_g(mean, cov, allow_singular=True)
            NLL_df['self'] += list(-gaussian.logpdf(components_subset))

            for col, nbr in row.iteritems():
                mean, cov = mc_params[nbr]
                gaussian = mv_g(mean, cov, allow_singular=True)
                NLL_df[col] += list(-gaussian.logpdf(components_subset))
    NLL_df = pd.DataFrame(NLL_df).set_index('cell_id')
    return neighbours, NLL_df

def strip_right(df, suffix='_nll'):
    """
    Remove suffix from all column names.
    """
    df.columns = df.columns.str.rstrip(suffix)
    
def filter_neighbours(nbrs, nlls, cluster='celltype'):
    """
    Restrict to neighbours matching cluster type of metacell. 
    All distances/nlls corresponding to neighbours of a different cluster type
    replaced with np.nan
    """
    
    distance = nbrs['Distance'].mask(~nbrs[f'{cluster}_match'])
    nbrs_filtered = nbrs.copy()
    nbrs_filtered['Distance'] = distance

    nbr_nlls = nlls.drop(['self', 'train'], axis=1).merge(nbrs[f'{cluster}_match'], left_on='mc_id', right_index=True, suffixes=('_nll','_nbr'))
    nlls_only = nbr_nlls.drop(['mc_id']+[x for x in nbr_nlls.columns if 'nbr' in str(x)], axis=1)
    nbrs_only = nbr_nlls.drop(['mc_id']+[x for x in nbr_nlls.columns if 'nll' in str(x)], axis=1)

    mask_vals = ~nbrs_only.values
    nlls_only = nlls_only.mask(mask_vals)
    strip_right(nlls_only)
    nlls_only.columns = nlls_only.columns.astype(int)
    nlls_f = nlls[['self','train','mc_id']].join(nlls_only)
    
    return nbrs_filtered, nlls_f

def plot_separation(nlls, 
                    nbr=1, 
                    save_as=None, 
                    title='Separation - NLL Difference between Assigned Metacell and Neighbour Metacell.',
                    xlim=99):
    plt.figure(figsize=(10,10))
    df = nlls.copy()
    df['diff'] = df[nbr]-df['self']
    xlim = np.nanpercentile(df['diff'], xlim)

    sns.ecdfplot(df.groupby('mc_id').mean()['diff'])
    plt.xlabel(f'NLL(nbr={nbr})-NLL(assigned MC)')
    plt.title(title)
    plt.xlim(None, xlim)
    
    plt.show()
    if save_as is not None:
        plt.savefig(save_as)
    plt.close()
    
def plot_compactness(nlls, 
                     save_as=None, 
                     title='Compactness - NLL for Assigned Metacell', 
                     xlim=100):
    plt.figure(figsize=(10,10))
    xlim = np.nanpercentile(nlls['self'], xlim)
    sns.ecdfplot(nlls.groupby('mc_id').mean()['self'])
    plt.xlabel(f'NLL(assigned MC)')
    plt.title(title)
    plt.xlim(None, xlim)
    
    plt.show()
    if save_as is not None:
        plt.savefig(save_as)
    plt.close()