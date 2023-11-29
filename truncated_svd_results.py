import tracemalloc
import time
import scanpy as sc
import pandas as pd

import cupy as cp
import cupyx
import numpy as np
from tqdm import tqdm

from sklearn.decomposition import TruncatedSVD

import sys

from SEACells.core import SEACells

def run_with_truncated_svd(ad, num_cells, use_gpu, use_sparse, threshold): 
    # In this case, we use truncated SVD to compute the kernel matrix 

    # User defined parameters 

    # Core parameters
    # number of SEACells
    n_SEACells = num_cells // 75
    build_kernel_on = 'X_pca' # key in ad.obsm to use for computing metacells
                            # This would be replaced by 'X_svd' for ATAC data
    
    # Additional parameters
    n_waypoint_eigs = 10 # Number of eigenvalues to consider when initializing metacells

    model = SEACells(ad, 
                     use_gpu=use_gpu,
                     use_sparse=use_sparse,
                     build_kernel_on=build_kernel_on,
                     n_SEACells=n_SEACells,
                     n_waypoint_eigs=n_waypoint_eigs,
                     convergence_epsilon = 1e-5)
    
    model.construct_kernel_matrix()
    M = model.M

    truncated_svd = TruncatedSVD(n_components=num_cells) 
    M_transformed = truncated_svd.fit_transform(M) 

    y = np.cumsum(truncated_svd.explained_variance_ratio_) 
    # Counts as the number of PCs that explain (threshold)% of the variance 
    n_components = np.where(y >= threshold)[0][0] + 1 

    truncated_svd = TruncatedSVD(n_components=n_components) 
    M_transformed = truncated_svd.fit_transform(M)
    

