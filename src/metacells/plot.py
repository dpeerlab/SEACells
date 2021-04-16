import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
sns.set_style("white")

import pandas as pd


def plot_2D(ad, key='X_umap', colour_metacells=True, cmap='Set2', title='Metacell Assignments'):
    """
    Plot 2D visualization of metacells.
    :param ad: annData containing 'Metacells' label in .obs
    :param key: (str) 2D embedding of data. Default: 'X_umap'
    :param colour_metacells: (bool) whether to colour cells by metacell assignment. Default: True
    :param cmap: matplotlib colormap for metacells. Default: 'Set2'
    :return:
        None
    """
    umap = pd.DataFrame(ad.obsm[key]).set_index(ad.obs_names).join(ad.obs['Metacell'])
    mcs = umap.loc[ad.obs['Metacell'].unique()]

    plt.figure(figsize=(10, 10))
    if colour_metacells:
        sns.scatterplot(x=0, y=1,
                        hue='Metacell',
                        data=umap,
                        s=50,
                        cmap=cmap,
                        legend=None)
        sns.scatterplot(x=0, y=1, s=150,
                        marker='v',
                        hue='Metacell',
                        data=mcs,
                        cmap=cmap,
                        edgecolor='black', linewidth=2,
                        legend=None)
    else:
        sns.scatterplot(x=0, y=1,
                        color='grey',
                        data=umap,
                        s=50,
                        cmap=cmap,
                        legend=None)
        sns.scatterplot(x=0, y=1, s=150,
                        marker='v',
                        color='red',
                        data=mcs,
                        cmap=cmap,
                        edgecolor='black', linewidth=2,
                        legend=None)

    plt.xlabel(f'{key}-0')
    plt.ylabel(f'{key}-1')
    plt.title(title)
    plt.show()
    plt.close()

def plot_metacell_sizes(ad, save_as=None, title='Distribution of Metacell Sizes'):
    """
    Plot distribution of cells contained per metacell.
    :param ad: annData containing 'Metacells' label in .obs
    :return: None
    """

    assert 'Metacell' in ad.obs, 'AnnData must contain "Metacell" in obs DataFrame.'
    label_df = ad.obs[['Metacell']].reset_index()
    plt.figure(figsize=(10, 10))
    sns.distplot(label_df.groupby('Metacell').count().iloc[:, 0])
    plt.xlabel('Number of Cells per Metacell')
    plt.title(title)
    plt.show()
    if save_as is not None:
        plt.savefig(save_as)
    plt.close()

    return