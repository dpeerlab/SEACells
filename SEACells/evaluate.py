import scanpy as sc
import anndata
import palantir
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def diffusion_component_variance(ad, low_dim_embedding):

    import palantir

    components = pd.DataFrame(ad.obsm[low_dim_embedding]).set_index(ad.obs_names)
    dm_res = palantir.utils.run_diffusion_maps(components)
    dc = palantir.utils.determine_multiscale_space(dm_res, n_eigs=10)

    return pd.DataFrame(dc.join(ad.obs["SEACell"]).groupby("SEACell").var().mean(1))


def diffusion_component_dist_to_NN(ad,
                                   low_dim_embedding,
                                   nth_nbr=1,
                                   cluster=None):
    components = pd.DataFrame(ad.obsm[low_dim_embedding]).set_index(ad.obs_names)
    dm_res = palantir.utils.run_diffusion_maps(components)
    dc = palantir.utils.determine_multiscale_space(dm_res, n_eigs=10)

    # Compute DC per metacell
    metacells_dcs = dc.join(ad.obs["SEACell"], how='inner').groupby("SEACell").mean()

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
        clusters = ad.obs.groupby("SEACell")[cluster].agg(lambda x: x.value_counts().index[0])
        nbr_clusters = pd.DataFrame(clusters.values[nbrs]).set_index(clusters.index)
        nbr_clusters.columns = metacells_nbrs.columns
        nbr_clusters = nbr_clusters.join(pd.DataFrame(clusters))

        clusters_match = nbr_clusters.eq(nbr_clusters[cluster], axis=0)
        return dists[nth_nbr][clusters_match[nth_nbr]]
    else:
        return dists[nth_nbr]


def get_density(ad, key, nth_neighbor=150):
    """
    Compute cell density as 1/ the distance to the 150th (by default) nearest neighbour.

    :param ad: AnnData object
    :param key: (str) key in ad.obsm to use to build diffusion components on.
    :param nth_neighbor:
    :return: pd.DataFrame containing cell ID and density.
    """
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=nth_neighbor)

    if 'key' in ad.obsm:
        print(f'Using {key} to compute cell density')
        components = pd.DataFrame(ad.obsm['X_pca']).set_index(ad.obs_names)
    else:
        raise ValueError(f'Key {key} not present in ad.obsm.')

    diffusion_map_results = palantir.utils.run_diffusion_maps(components)
    diffusion_components = palantir.utils.determine_multiscale_space(diffusion_map_results, n_eigs=8)

    nbrs = neigh.fit(diffusion_components)
    cell_density = pd.DataFrame(nbrs.kneighbors()[0][:, nth_neighbor - 1]).set_index(ad.obs_names).rename(
        columns={0: 'density'})
    density = 1 / cell_density

    return density


def celltype_frac(x, col_name):
    val_counts = x[col_name].value_counts()
    return val_counts.values[0] / val_counts.values.sum()


def compute_celltype_purity(ad, col_name):
    """
    Compute the purity (prevalence of most abundant value) of the specified col_name from ad.obs within each metacell.
    @param: ad - AnnData object with SEACell assignment and col_name in ad.obs dataframe
    @param: col_name - (str) column name within ad.obs representing celltype groupings for each cell.
    """

    celltype_fraction = ad.obs.groupby('SEACell').apply(lambda x: celltype_frac(x, col_name))
    celltype = ad.obs.groupby('SEACell').apply(lambda x: x[col_name].value_counts().index[0])

    return pd.concat([celltype, celltype_fraction], axis=1).rename(columns={0: col_name, 1: f'{col_name}_purity'})




