import tracemalloc
import time
import scanpy as sc
import pandas as pd

import cupy as cp
import cupyx
import numpy as np
from tqdm import tqdm

import sys

# from icecream import ic

# from importlib import reload
from SEACells.core import SEACells

# reload(SEACells)

# num_cells = 10000
# ad = ad[:num_cells]


def get_data(ad, num_cells, use_gpu, use_sparse, A_init = None, B_init = None, K_init = None):
    ## User defined parameters

    ## Core parameters
    # number of SEACells
    n_SEACells = num_cells // 75
    build_kernel_on = "X_pca"  # key in ad.obsm to use for computing metacells
    # This would be replaced by 'X_svd' for ATAC data

    ## Additional parameters
    n_waypoint_eigs = (
        10  # Number of eigenvalues to consider when initializing metacells
    )

    model = SEACells(
        ad,
        use_gpu=use_gpu,
        use_sparse=use_sparse,
        build_kernel_on=build_kernel_on,
        n_SEACells=n_SEACells,
        n_waypoint_eigs=n_waypoint_eigs,
        convergence_epsilon=1e-5,
    )
    if K_init is None:
        model.construct_kernel_matrix()
    else: 
        model.add_precomputed_kernel_matrix(K_init)
    model.initialize_archetypes()
    model.initialize() 

    if A_init is not None:
        model.A_ = A_init 
    if B_init is not None:
        model.B_ = B_init

    start = time.time()
    tracemalloc.start()

    model.fit(min_iter=10, max_iter=150)

    end = time.time()
    tot_time = end - start

    mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assignments = model.get_hard_assignments()

    # # Get the final A and B matrices
    # A = model.A_
    # B = model.B_
    # K = model.kernel_matrix

    # #   Get the sparsity dataframe
    # sparsity = model.sparsity_ratios

    return assignments, tot_time, mem


def gpu_versions(ad, num_cells):
    # assignments2, time2, mem2, A_init, B_init, K_init, sparsity = get_data(
    #         ad, num_cells=num_cells, use_gpu=False, use_sparse=True
    #     )

    try: 
        assignments, time, mem = get_data(ad, num_cells=num_cells, use_gpu=False, use_sparse=False)
        # write the assignments to a csv file
        assignments.to_csv(f"results/{num_cells}_cells/assignments_v1.csv")
    except: 
        pass 


def get_results(num_cell):
    #    potential_num_cells = [5000, 10000, 50000, 100000, 150000, 200000]
    #    for num_cell in potential_num_cells:
    ad = sc.read("/home/aparna/DATA/aparnakumar/150000_cells/mouse_marioni_150k.h5ad")
    ad = ad[:num_cell]
    for trial in range(1):
        gpu_versions(ad, num_cell)

        # assignments, comparisons = gpu_versions(ad, num_cell)

        # try:
        #     comparisons.to_csv(f"results/{num_cell}_cells/comparisons_{trial}.csv")
        # except:
        #     pass

        # for i in range(len(assignments)):
        #     if i == 0:
        #         try: 
        #             assignments[i].to_csv(f"results/{num_cell}_cells/assignments_v2_{trial}.csv")
        #         except:
        #             pass
        #     elif i == 1:
        #         try: 
        #             assignments[i].to_csv(f"results/{num_cell}_cells/assignments_v3_{trial}.csv")
        #         except:
        #             pass
        #     elif i == 2:
        #         try: 
        #             assignments[i].to_csv(f"results/{num_cell}_cells/assignments_v4_{trial}.csv")
        #         except:
        #             pass
                

        print(f"Done with {num_cell} cells, trial {trial + 1}")


# Create main function
if __name__ == "__main__":
    # Runs get_data based on the num_cells given as command line input
    num_cells = int(sys.argv[1])
    # print(type(num_cells))
    get_results(num_cells)
