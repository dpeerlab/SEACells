def log_transform(ad, ps=0.1):
    ad.X = np.log2(ad.X + ps) - np.log2(ps)

def subset_anndata(init_ad):
    import anndata 
    
    try:
        ad = anndata.AnnData(init_ad.raw.X)
        ad.raw = ad
        ad.obs_names = init_ad.obs_names
        ad.var_names = init_ad.var_names
        ad.obs = init_ad.obs

        # Run PCA and UMAP etc on new abridged anndata
        sc.pp.normalize_per_cell(ad)
        log_transformtransform(ad)

        sc.pp.highly_variable_genes(ad, flavor='cell_ranger', n_top_genes=2500)
        sc.pp.pca(ad, use_highly_variable=True, n_comps=50)
        sc.pp.neighbors(ad)
        sc.tl.umap(ad)

    except:
        print('Skipping raw data and normalization; copying data directly...')
        ad = init_ad
    
    return ad 