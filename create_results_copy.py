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
from SEACells.core_copy import SEACells

# reload(SEACells)

# num_cells = 10000
# ad = ad[:num_cells]


def get_data(ad, num_cells, use_gpu, use_sparse, A_init=None, B_init=None, K_init=None, l1_penalty = 0):
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
    model.l1_penalty = l1_penalty
    if K_init is None:
        model.construct_kernel_matrix()
    else:
        model.add_precomputed_kernel_matrix(K_init)
    # model.initialize_archetypes()
    # model.initialize()

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

    # Get the final A and B matrices
    A = model.A_
    B = model.B_
    K = model.kernel_matrix

    #   Get the sparsity dataframe
    sparsity = model.sparsity_ratios

    return assignments, tot_time, mem, A, B, K, sparsity


def gpu_versions(ad, num_cells):
    # assignments2, time2, mem2, A_init, B_init, K_init, sparsity = get_data(
    #     ad, num_cells=num_cells, use_gpu=False, use_sparse=True
    # )
    # Clear the GPU memory
    cp.get_default_memory_pool().free_all_blocks() 
    cp.get_default_pinned_memory_pool().free_all_blocks()

    # try:
    #     assignments2, time2, mem2, A, B, K, sparsity = get_data(
    #         ad,
    #         num_cells=num_cells,
    #         use_gpu=False,
    #         use_sparse=True,
    #         # A_init=cupyx.scipy.sparse.csr_matrix(A_init),
    #         # B_init=cupyx.scipy.sparse.csr_matrix(B_init),
    #         # K_init=cupyx.scipy.sparse.csr_matrix(K_init),
    #     )
    #     # If successful, write the time and memory a file "{num_cells}_cells/v2_{timestamp}.txt"
    #     # Get the timestamp as a number
    #     timestamp = time.time()

    #     # Write the time and memory data
    #     with open(f"results14-files_in_copy/{num_cells}_cells/v2_{timestamp}.txt", "w") as f:
    #         f.write(f"Time: {time2}\n")
    #         f.write(f"Memory: {mem2}\n")

    #     # If assignments is not None, write it to a file
    #     if assignments2 is not None:
    #         assignments2.to_csv(
    #             f"results14-files_in_copy/{num_cells}_cells/assignments_v2_{timestamp}.csv"
    #         )

    #     # Write the A and B matrices
    #     np.save(f"results14-files_in_copy/{num_cells}_cells/A_v2_{timestamp}.npy", A)
    #     np.save(f"results14-files_in_copy/{num_cells}_cells/B_v2_{timestamp}.npy", B)

    #     # Write the sparsity dataframe
    #     sparsity.to_csv(f"results14-files_in_copy/{num_cells}_cells/sparsity_v2_{timestamp}.csv")

    #     # Clear the GPU memory
    #     cp.get_default_memory_pool().free_all_blocks()

    # except Exception as e:
    #     # fill with nans if it fails
    #     assignments2, time2, mem2 = ([np.nan, np.nan, np.nan], np.nan, [np.nan, np.nan])

    #     # If it fails, write the error to a file "{num_cells}_cells/v2_{timestamp}.txt"
    #     # Get the timestamp as a number
    #     timestamp = time.time()

    #     # Write the error to a file
    #     with open(f"results14-files_in_copy/{num_cells}_cells/v2_{timestamp}.txt", "w") as f:
    #         f.write(f"Error: {e}\n")

    #     # Clear the GPU memory
    #     cp.get_default_memory_pool().free_all_blocks()
    #     cp.get_default_pinned_memory_pool().free_all_blocks()

    try:
        assignments4, time4, mem4, A, B, K, sparsity = get_data(
            ad,
            num_cells=num_cells,
            use_gpu=True,
            use_sparse=True,
            # A_init=cupyx.scipy.sparse.csr_matrix(A_init),
            # B_init=cupyx.scipy.sparse.csr_matrix(B_init),
            # K_init=cupyx.scipy.sparse.csr_matrix(K_init),
        )
        # If successful, write the time and memory a file "{num_cells}_cells/v4_{timestamp}.txt"
        # Get the timestamp as a number
        timestamp = time.time()

        # Write the time and memory data
        with open(f"results14-files_in_copy/{num_cells}_cells/v4_{timestamp}.txt", "w") as f:
            f.write(f"Time: {time4}\n")
            f.write(f"Memory: {mem4}\n")

        # If assignments is not None, write it to a file
        if assignments4 is not None:
            assignments4.to_csv(
                f"results14-files_in_copy/{num_cells}_cells/assignments_v4_{timestamp}.csv"
            )

        # Write the A and B matrices
        np.save(f"results14-files_in_copy/{num_cells}_cells/A_v4_{timestamp}.npy", A)
        np.save(f"results14-files_in_copy/{num_cells}_cells/B_v4_{timestamp}.npy", B)

        # Write the sparsity dataframe
        sparsity.to_csv(f"results14-files_in_copy/{num_cells}_cells/sparsity_v4_{timestamp}.csv")

        # Clear the GPU memory
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    except Exception as e:
        # fill with nans if it fails
        assignments4, time4, mem4 = ([np.nan, np.nan, np.nan], np.nan, [np.nan, np.nan])

        # If it fails, write the error to a file "{num_cells}_cells/v4_{timestamp}.txt"
        # Get the timestamp as a number
        timestamp = time.time()

        # Write the error to a file
        with open(f"results14-files_in_copy/{num_cells}_cells/v4_{timestamp}.txt", "w") as f:
            f.write(f"Error: {e}\n")

        # Clear the GPU memory
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    # try: 
    #     assignments3, time3, mem3, A, B, K, sparsity = get_data(
    #         ad,
    #         num_cells=num_cells,
    #         use_gpu=True,
    #         use_sparse=False,
    #         # A_init=cupyx.scipy.sparse.csr_matrix(A_init),
    #         # B_init=cupyx.scipy.sparse.csr_matrix(B_init),
    #         # K_init=cupyx.scipy.sparse.csr_matrix(K_init),
    #     )
    #     # If successful, write the time and memory a file "{num_cells}_cells/v3_{timestamp}.txt"
    #     # Get the timestamp as a number
    #     timestamp = time.time()

    #     # Write the time and memory data
    #     with open(f"results14-files_in_copy/{num_cells}_cells/v3_{timestamp}.txt", "w") as f:
    #         f.write(f"Time: {time3}\n")
    #         f.write(f"Memory: {mem3}\n")

    #     # If assignments is not None, write it to a file
    #     if assignments3 is not None:
    #         assignments3.to_csv(
    #             f"results14-files_in_copy/{num_cells}_cells/assignments_v3_{timestamp}.csv"
    #         )

    #     # Write the A and B matrices
    #     np.save(f"results14-files_in_copy/{num_cells}_cells/A_v3_{timestamp}.npy", A)
    #     np.save(f"results14-files_in_copy/{num_cells}_cells/B_v3_{timestamp}.npy", B)

    #     # Write the sparsity dataframe
    #     sparsity.to_csv(f"results14-files_in_copy/{num_cells}_cells/sparsity_v3_{timestamp}.csv")

    #     # Clear the GPU memory
    #     cp.get_default_memory_pool().free_all_blocks()
    #     cp.get_default_pinned_memory_pool().free_all_blocks()

    # except Exception as e:
    #     # fill with nans if it fails
    #     assignments3, time3, mem3 = ([np.nan, np.nan, np.nan], np.nan, [np.nan, np.nan])

    #     # If it fails, write the error to a file "{num_cells}_cells/v3_{timestamp}.txt"
    #     # Get the timestamp as a number
    #     timestamp = time.time()

    #     # Write the error to a file
    #     with open(f"results14-files_in_copy/{num_cells}_cells/v3_{timestamp}.txt", "w") as f:
    #         f.write(f"Error: {e}\n")

    #     # Clear the GPU memory
    #     cp.get_default_memory_pool().free_all_blocks()
    #     cp.get_default_pinned_memory_pool().free_all_blocks()


    # # Write the assignments
    # assignments = [assignments2, assignments3, assignments4]

    # # Write the time and memory data
    # comparisons = pd.DataFrame({'version': ['v2: no GPU, yes sparse', 'v3: yes GPU, no sparse', 'v4: yes GPU, yes sparse'],
    #                        'time (s)': [time2, time3, time4],
    #                        'peak memory': [mem2[1], mem3[1], mem4[1]]})

    # return assignments, comparisons


def get_results(num_cell):
    #    potential_num_cells = [5000, 10000, 50000, 100000, 150000, 200000]
    #    for num_cell in potential_num_cells:
    ad = sc.read("/home/aparna/DATA/aparnakumar/150000_cells/mouse_marioni_150k.h5ad")
    ad = ad[:num_cell]
    # for l1_penalty in [0.1, 0.5, 1, 5]:
    for trial in range(3):
        # gpu_versions(ad, num_cell)

        assignments, comparisons = gpu_versions(ad, num_cell)

        try:
            comparisons.to_csv(f"results14-files_in_copy/{num_cell}_cells/comparisons_{trial}.csv")
        except:
            pass

        for i in range(len(assignments)):
            if i == 0:
                try:
                    assignments[i].to_csv(f"results14-files_in_copy/{num_cell}_cells/assignments_v2_{trial}.csv")
                except:
                    pass
            elif i == 1:
                try:
                    assignments[i].to_csv(f"results14-files_in_copy/{num_cell}_cells/assignments_v3_{trial}.csv")
                except:
                    pass
            elif i == 2:
                try:
                    assignments[i].to_csv(f"results14-files_in_copy/{num_cell}_cells/assignments_v4_{trial}.csv")
                except:
                    pass

        print(f"Done with {num_cell} cells, trial {trial + 1}")
        #print(f"Done with {num_cell} cells, l1_penalty {l1_penalty}")


# Create main function
if __name__ == "__main__":
    # Runs get_data based on the num_cells given as command line input
    num_cells = int(sys.argv[1])
    get_results(num_cells)
    # print(type(num_cells))
    # cells_list = [50000, 100000, 150000]
    # for num_cells in cells_list:
    #     get_results(num_cells)
