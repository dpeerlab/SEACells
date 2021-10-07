import scanpy as sc
import anndata
import palantir
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns 


ALG_NAME = "AEROCell"

def diffusion_component_variance(ad, 
                                low_dim_embedding):
    import palantir

    components = pd.DataFrame(ad.obsm[low_dim_embedding]).set_index(ad.obs_names)
    dm_res = palantir.utils.run_diffusion_maps(components)
    dc = palantir.utils.determine_multiscale_space(dm_res, n_eigs=10)

    return pd.DataFrame(dc.join(ad.obs[ALG_NAME]).groupby(ALG_NAME).var().mean(1))

def diffusion_component_dist_to_NN(ad, 
                              low_dim_embedding, 
                              nth_nbr=1,
                              cluster=None):
    
    components = pd.DataFrame(ad.obsm[low_dim_embedding]).set_index(ad.obs_names)
    dm_res = palantir.utils.run_diffusion_maps(components)
    dc = palantir.utils.determine_multiscale_space(dm_res, n_eigs=10)
    

    # Compute DC per metacell
    metacells_dcs = dc.join(ad.obs[ALG_NAME], how='inner').groupby(ALG_NAME).mean()

    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=nth_nbr)
    nbrs = neigh.fit(metacells_dcs)
    dists, nbrs = nbrs.kneighbors()
    dists = pd.DataFrame(dists).set_index(metacells_dcs.index)
    dists.columns += 1
    
    nbr_cells = np.array(metacells_dcs.index)[nbrs]

    metacells_nbrs = pd.DataFrame(nbr_cells)
    metacells_nbrs.index = metacells_dcs.index
    metacells_nbrs.columns += 1

    if cluster is not None:
        
        # Get cluster type of neighbors to ensure they match the metacell cluster
        clusters = ad.obs.groupby(ALG_NAME)[cluster].agg(lambda x:x.value_counts().index[0])
        nbr_clusters = pd.DataFrame(clusters.values[nbrs]).set_index(clusters.index)
        nbr_clusters.columns = metacells_nbrs.columns
        nbr_clusters = nbr_clusters.join(pd.DataFrame(clusters))

        clusters_match = nbr_clusters.eq(nbr_clusters[cluster], axis=0)
        return dists[nth_nbr][clusters_match[nth_nbr]]
    else:
        return dists[nth_nbr]
    

def get_density(ad, nth_neighbor=150):
    """
    Compute cell density as 1/ the distance to the 150th (by default) nearest neighbour
    Modifies the ad.uns['metacell_metrics'] dictionary to add a key 'density' which contains
    output of this function.

    :param ad:
    :param nth_neighbor:
    :return: pd.DataFrame containing the following columns:
        ALG_NAME - Metacell label for each cell
        'density' - density of each cell
        'tertile' - which tertile (0=lowers, 2=highest) each cell belongs to in terms of individual density
        'mc_density' - average density of the metacell each cell belongs to
        'mc_density_tertile' - which tertile (0=lowers, 2=highest) each cell belongs to in terms of metacell density
    """
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=nth_neighbor)
    
    if 'X_pca' in ad.obsm:
        print('Using PCA')
        pca_components = pd.DataFrame(ad.obsm['X_pca']).set_index(ad.obs_names)
    else:
        print('Using SVD')
        pca_components = pd.DataFrame(ad.obsm['X_svd']).set_index(ad.obs_names)
   
        
    dm_res = palantir.utils.run_diffusion_maps(pca_components)
    dc = palantir.utils.determine_multiscale_space(dm_res, n_eigs=8)

    nbrs = neigh.fit(dc)
    cell_density = pd.DataFrame(nbrs.kneighbors()[0][:,nth_neighbor-1]).set_index(ad.obs_names).rename(columns={0:'density'})
    density = 1/cell_density
    
    density['tertile'] = pd.qcut(density['density'], 3, labels=False)
    density = ad.obs.join(density)[[ALG_NAME,'density','tertile']]
    
    mc_density = density.groupby(ALG_NAME).mean()[['density']].rename(columns={'density':'mc_density'})
    mc_density['mc_density_tertile'] = pd.qcut(mc_density['mc_density'], 3, labels=False)
    
    density = density.merge(mc_density, left_on=ALG_NAME, right_index=True)

    return density

def celltype_frac(x, col_name):
    val_counts = x[col_name].value_counts()
    return val_counts.values[0]/val_counts.values.sum()

def compute_within_metacell_purity(ad, col_name):
    """
    Compute the purity (prevalence of most abundant value) of the specified col_name from ad.obs within each metacell. 
    @param: ad - AnnData object with ALG_NAME assignment and col_name in ad.obs dataframe
    @param: col_name - (str) column name within ad.obs. Usually, some type of cluster or 'celltype' or similar
    """

    celltype_fraction = ad.obs.groupby(ALG_NAME).apply(lambda x: celltype_frac(x, col_name))
    celltype =  ad.obs.groupby(ALG_NAME).apply(lambda x: x[col_name].value_counts().index[0])

    return pd.concat([celltype, celltype_fraction], axis=1).rename(columns={0:col_name, 1: f'{col_name}_purity'})

def compute_within_metacell_entropy(ad, col_name):
    """
    Compute the entropy of the specified col_name from ad.obs within each metacell. 
    @param: ad - AnnData object with ALG_NAME assignment and col_name in ad.obs dataframe
    @param: col_name - (str) column name within ad.obs. Usually, some type of cluster or 'celltype' or similar
    """
    from collections import Counter
    from scipy.stats import entropy
    mc_ids = []
    mc_entropy = []
    for mc in ad.obs[ALG_NAME].unique():
        mc_ids.append(mc)
        cell_types = Counter(ad.obs[ad.obs[ALG_NAME]==mc][col_name])

        mc_entropy.append(entropy(list(cell_types.values())))
    
    entropies = pd.DataFrame(mc_entropy)
    entropies.columns = ['entropy']
    entropies.index = mc_ids

    return entropies



