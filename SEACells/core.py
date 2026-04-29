import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from . import evaluate
except ImportError:
    import evaluate


def SEACells(
    ad,
    build_kernel_on: str,
    n_SEACells: int,
    use_gpu: bool = False,
    verbose: bool = True,
    n_waypoint_eigs: int = 10,
    n_neighbors: int = 15,
    convergence_epsilon: float = 1e-3,
    l2_penalty: float = 0,
    max_franke_wolfe_iters: int = 50,
    use_sparse: bool = False,
):
    """Core SEACells class.

    :param ad: (AnnData) annotated data matrix
    :param build_kernel_on: (str) key corresponding to matrix in ad.obsm which is used to compute kernel for metacells
                            Typically 'X_pca' for scRNA or 'X_svd' for scATAC
    :param n_SEACells: (int) number of SEACells to compute
    :param use_gpu: (bool) whether to use GPU for computation
    :param verbose: (bool) whether to suppress verbose program logging
    :param n_waypoint_eigs: (int) number of eigenvectors to use for waypoint initialization
    :param n_neighbors: (int) number of nearest neighbors to use for graph construction
    :param convergence_epsilon: (float) convergence threshold for Franke-Wolfe algorithm
    :param l2_penalty: (float) L2 penalty for Franke-Wolfe algorithm
    :param max_franke_wolfe_iters: (int) maximum number of iterations for Franke-Wolfe algorithm
    :param use_sparse: (bool) whether to use sparse matrix operations. Currently only supported for CPU implementation.

    See cpu.py or gpu.py for descriptions of model attributes and methods.
    """
    if use_sparse:
        assert (
            not use_gpu
        ), "Sparse matrix operations are only supported for CPU implementation."
        try:
            from . import cpu
        except ImportError:
            import cpu
        model = cpu.SEACellsCPU(
            ad,
            build_kernel_on,
            n_SEACells,
            verbose,
            n_waypoint_eigs,
            n_neighbors,
            convergence_epsilon,
            l2_penalty,
            max_franke_wolfe_iters,
        )

        return model

    if use_gpu:
        try:
            from . import gpu
        except ImportError:
            import gpu

        model = gpu.SEACellsGPU(
            ad,
            build_kernel_on,
            n_SEACells,
            verbose,
            n_waypoint_eigs,
            n_neighbors,
            convergence_epsilon,
            l2_penalty,
            max_franke_wolfe_iters,
        )

    else:
        try:
            from . import cpu_dense
        except ImportError:
            import cpu_dense
        model = cpu_dense.SEACellsCPUDense(
            ad,
            build_kernel_on,
            n_SEACells,
            verbose,
            n_waypoint_eigs,
            n_neighbors,
            convergence_epsilon,
            l2_penalty,
            max_franke_wolfe_iters,
        )

    return model


def sparsify_assignments(A, thresh: float):
    """Zero out all values below a threshold in an assignment matrix.

    :param A: (csr_matrix) of shape n_cells x n_SEACells containing assignment weights
    :param thresh: (float) threshold below which to zero out assignment weights
    :return: (np.array) of shape n_cells x n_SEACells containing assignment weights.
    """
    A = copy.deepcopy(A)
    A[A < thresh] = 0

    # Renormalize. Cells whose every weight was below threshold would otherwise
    # produce NaN rows; leave their weights at zero so they contribute nothing.
    row_sums = A.sum(1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    A = A / row_sums

    return A


def summarize_by_soft_SEACell(
    ad, A, celltype_label=None, summarize_layer="raw", minimum_weight: float = 0.05
):
    """Summary of soft SEACell assignment.

    Aggregates cells within each SEACell, summing over all raw data x assignment weight for all cells belonging to a
    SEACell. Data is un-normalized and pseudo-raw aggregated counts are stored in .layers['raw'].
    Attributes associated with variables (.var) are copied over, but relevant per SEACell attributes must be
    manually copied, since certain attributes may need to be summed, or averaged etc, depending on the attribute.
    The output of this function is an anndata object of shape n_metacells x original_data_dimension.

    @param ad: (sc.AnnData) containing raw counts for single-cell data
    @param A: (np.array) of shape n_SEACells x n_cells containing assignment weights of cells to SEACells
    @param celltype_label: (str) optionally provide the celltype label to compute modal celltype per SEACell
    @param summarize_layer: (str) key for ad.layers to find raw data. Use 'raw' to search for ad.raw.X
    @param minimum_weight: (float) minimum value below which assignment weights are zero-ed out. If all cell assignment
                            weights are smaller than minimum_weight, the 95th percentile weight is used.
    @return: aggregated anndata containing weighted expression for aggregated SEACells
    """
    import scanpy as sc
    from scipy.sparse import csr_matrix

    compute_seacell_celltypes = False
    if celltype_label is not None:
        if celltype_label not in ad.obs.columns:
            raise ValueError(f"Celltype label {celltype_label} not present in ad.obs")
        compute_seacell_celltypes = True

    if summarize_layer == "raw" and ad.raw is not None:
        data = ad.raw.X
    else:
        data = ad.layers[summarize_layer]

    A = sparsify_assignments(A.T, thresh=minimum_weight)

    # Vectorized aggregation: a single sparse matmul replaces the per-metacell
    # Python loop. Mathematically the same as
    #     seacell_exp[m, :] = sum_c A[c, m] * data[c, :] / sum_c A[c, m]
    # but expressed as (A.T @ data) / totals.
    n_metacells = A.shape[1]
    # Per-metacell weight totals; works uniformly for dense ndarray and sparse.
    totals = np.asarray(A.sum(axis=0)).ravel()
    totals_safe = np.where(totals > 0, totals, 1.0)

    A_T = csr_matrix(A.T)  # (M, n_cells)
    weighted_sum = A_T @ data  # (M, n_features)
    if hasattr(weighted_sum, "toarray"):
        weighted_sum_dense = weighted_sum.toarray()
    else:
        weighted_sum_dense = np.asarray(weighted_sum)
    seacell_expressions_mat = weighted_sum_dense / totals_safe[:, np.newaxis]
    # Rows whose total weight was zero produce all zeros (totals_safe = 1
    # divides a zero numerator), matching the per-metacell zero fallback.

    seacell_expressions = csr_matrix(seacell_expressions_mat)
    seacell_ad = sc.AnnData(seacell_expressions, dtype=seacell_expressions.dtype)
    seacell_ad.var_names = ad.var_names
    seacell_ad.obs["Pseudo-sizes"] = totals

    if compute_seacell_celltypes:
        # Vectorized celltype purity: build a cells x celltypes indicator and
        # form purity = A.T @ indicator (M x C). The dominant celltype per
        # metacell is argmax over rows; ties resolve to the first category, as
        # in the original sort_values(...).iloc[0] path (categories are sorted).
        celltype_col = ad.obs[celltype_label].astype("category")
        celltype_codes = celltype_col.cat.codes.values
        celltype_names = celltype_col.cat.categories
        n_cells = len(celltype_codes)
        celltype_indicator = csr_matrix(
            (np.ones(n_cells), (np.arange(n_cells), celltype_codes)),
            shape=(n_cells, len(celltype_names)),
        )
        purity_mat = A_T @ celltype_indicator  # (M, C)
        if hasattr(purity_mat, "toarray"):
            purity_mat = purity_mat.toarray()
        purity_mat = np.asarray(purity_mat)

        purity_row_sum = purity_mat.sum(axis=1)
        nonempty = purity_row_sum > 0
        # Avoid divide-by-zero rows; we'll mask their celltype to None below.
        denom = np.where(nonempty, purity_row_sum, 1.0)
        purity_norm = purity_mat / denom[:, np.newaxis]
        argmax_ct = purity_norm.argmax(axis=1)

        seacell_celltypes = [
            celltype_names[idx] if nonempty[m] else None
            for m, idx in enumerate(argmax_ct)
        ]
        seacell_purities = np.where(
            nonempty, purity_norm[np.arange(n_metacells), argmax_ct], 0.0
        ).tolist()

        seacell_ad.obs["celltype"] = seacell_celltypes
        seacell_ad.obs["celltype_purity"] = seacell_purities
    seacell_ad.var_names = ad.var_names
    return seacell_ad


def summarize_by_SEACell(
    ad, SEACells_label="SEACell", celltype_label=None, summarize_layer="raw"
):
    """Summary of SEACell assignment.

    Aggregates cells within each SEACell, summing over all raw data for all cells belonging to a SEACell.
    Data is unnormalized and raw aggregated counts are stored .layers['raw'].
    Attributes associated with variables (.var) are copied over, but relevant per SEACell attributes must be
    manually copied, since certain attributes may need to be summed, or averaged etc, depending on the attribute.
    The output of this function is an anndata object of shape n_metacells x original_data_dimension.
    :return: anndata.AnnData containing aggregated counts.

    """
    import scanpy as sc
    from scipy.sparse import csr_matrix

    # Pick the source data matrix once.
    if summarize_layer == "X":
        data = ad.X
    elif summarize_layer == "raw" and ad.raw is not None:
        data = ad.raw.X
    else:
        data = ad.layers[summarize_layer]

    # Build a cell-to-metacell indicator and aggregate in one sparse matmul.
    # Preserves first-occurrence order of metacell labels (matches the prior
    # use of pd.Series.unique() to seed summ_matrix.index).
    labels = ad.obs[SEACells_label]
    metacell_order = pd.Index(labels.unique())
    code_lookup = pd.Series(np.arange(len(metacell_order)), index=metacell_order)
    codes = code_lookup.loc[labels.values].values
    n_cells = len(codes)
    n_metacells = len(metacell_order)

    indicator = csr_matrix(
        (np.ones(n_cells), (codes, np.arange(n_cells))),
        shape=(n_metacells, n_cells),
    )
    summed = indicator @ data
    summed = csr_matrix(summed)

    meta_ad = sc.AnnData(summed, dtype=summed.dtype)
    meta_ad.obs_names = metacell_order.astype(str)
    meta_ad.var_names = ad.var_names
    meta_ad.layers["raw"] = summed

    # Also compute cell type purity
    if celltype_label is not None:
        # TODO: Catch specific exception
        try:
            purity_df = evaluate.compute_celltype_purity(ad, celltype_label)
            meta_ad.obs = meta_ad.obs.join(purity_df)
        except Exception as e:  # noqa: BLE001
            print(f"Cell type purity failed with Exception {e}")

    return meta_ad
