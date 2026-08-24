"""Unified CPU/GPU implementation of the SEACells kernel-archetypal-analysis algorithm.

A single :class:`SEACellsModel` holds all algorithm and orchestration logic. The only
differences between the CPU and GPU code paths are:

* ``self.xp`` / ``self.sp`` -- the array + sparse modules (numpy/scipy vs cupy/cupyx),
  which cover ~90% of the work (Frank-Wolfe updates, RSS, greedy init, kernel arithmetic);
* a small number of explicitly-marked backend hooks for operations that need different
  *libraries* rather than a different array module (exact kNN, diffusion-map waypoints,
  host/device transfer).

This removes the duplication (and the silent CPU/GPU drift bugs) of the old
``cpu_dense`` / ``cpu`` / ``gpu`` modules while keeping one algorithm implementation.

Design notes
------------
* On the GPU path the kernel ``M``, its Gram matrix ``K = M @ M.T``, and the ``A``/``B``
  weight matrices stay resident on the device for the whole fit; only the final
  assignments are copied back to the host.
* RSS is computed in a memory-scalable reduced form (``O(n*s)`` memory) rather than by
  materializing the ``n x n`` reconstruction, which is infeasible on the GPU at scale.
* Exact kNN is the default on both backends (reproducible, and a strict accuracy upgrade
  over the previous approximate scanpy/pynndescent path). Approximate ANN can be added as
  an opt-in for very large ``n``.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


class SEACellsModel:
    """Unified kernel-archetypal-analysis metacell solver (CPU or GPU)."""

    def __init__(
        self,
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
        dtype=np.float32,
    ):
        """Create a SEACells model.

        :param ad: (AnnData) annotated data matrix.
        :param build_kernel_on: (str) key in ``ad.obsm`` used to build the kernel
            (``'X_pca'`` for scRNA, ``'X_svd'`` for scATAC).
        :param n_SEACells: (int) number of metacells (archetypes) to compute.
        :param use_gpu: (bool) run on GPU (cupy/cuml) if True, else CPU (numpy/scipy).
        :param verbose: (bool) verbose logging.
        :param n_waypoint_eigs: (int) number of eigenvectors for waypoint initialization.
        :param n_neighbors: (int) number of nearest neighbors for graph construction.
        :param convergence_epsilon: (float) convergence threshold multiplier on initial RSS.
        :param l2_penalty: (float) L2 penalty in the A update.
        :param max_franke_wolfe_iters: (int) Frank-Wolfe inner iterations for A and B.
        :param dtype: numpy dtype used for kernel construction (default float32).
        """
        print("Welcome to SEACells!" + (" [GPU]" if use_gpu else ""))
        self.ad = ad
        self.build_kernel_on = build_kernel_on
        self.n_cells = ad.shape[0]

        if not isinstance(n_SEACells, int):
            try:
                n_SEACells = int(n_SEACells)
            except ValueError:
                raise ValueError(
                    f"The number of SEACells specified must be an integer type, not {type(n_SEACells)}"
                )
        self.k = n_SEACells

        self.use_gpu = use_gpu
        self.dtype = dtype
        if use_gpu:
            import cupy as cp
            import cupyx.scipy.sparse as csp

            self.xp = cp
            self.sp = csp
        else:
            import scipy.sparse as ssp

            self.xp = np
            self.sp = ssp

        self.n_waypoint_eigs = n_waypoint_eigs
        self.waypoint_proportion = 1
        self.n_neighbors = n_neighbors

        self.max_FW_iter = max_franke_wolfe_iters
        self.verbose = verbose
        self.l2_penalty = l2_penalty

        self.RSS_iters = []
        self.convergence_epsilon = convergence_epsilon
        self.convergence_threshold = None

        self.kernel_matrix = None  # M  (n x n, backend sparse)
        self.K = None  # M @ M.T  (n x n, backend sparse)
        self._Mnorm2 = None  # ||M||_F^2 cached for reduced RSS

        self.archetypes = None
        self.A_ = None
        self.B_ = None
        self.B0 = None

    # ------------------------------------------------------------------ #
    # Backend hooks (the only library-level CPU/GPU differences)
    # ------------------------------------------------------------------ #
    def _to_host(self, x):
        """Return a numpy version of a backend array (no-op on CPU)."""
        if self.use_gpu:
            return self.xp.asnumpy(x)
        return np.asarray(x)

    def _knn(self, X, k):
        """Exact k-nearest-neighbors on the active backend.

        :param X: (n, d) embedding (numpy array).
        :param k: number of neighbors (includes self).
        :return: (dist, idx) as backend arrays of shape (n, k); Euclidean distances.
        """
        if self.use_gpu:
            from cuml.neighbors import NearestNeighbors

            Xb = self.xp.asarray(X, dtype=self.dtype)
            nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
            nn.fit(Xb)
            dist, idx = nn.kneighbors(Xb)
            return self.xp.asarray(dist), self.xp.asarray(idx)
        else:
            from sklearn.neighbors import NearestNeighbors

            Xb = np.asarray(X, dtype=self.dtype)
            nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
            nn.fit(Xb)
            dist, idx = nn.kneighbors(Xb)
            return np.asarray(dist), np.asarray(idx)

    # ------------------------------------------------------------------ #
    # Kernel construction
    # ------------------------------------------------------------------ #
    def add_precomputed_kernel_matrix(self, K):
        """Provide a precomputed kernel matrix ``M`` (moves it to the active backend)."""
        assert K.shape == (self.n_cells, self.n_cells), (
            f"Dimension of kernel matrix must be n_cells = "
            f"({self.n_cells},{self.n_cells}), not {K.shape}"
        )
        M = self.sp.csr_matrix(K)
        self.kernel_matrix = M
        self.K = (M @ M.T).tocsr()
        self._Mnorm2 = float(self._to_host(M.multiply(M).sum()))

    def construct_kernel_matrix(self, n_neighbors: int = None, graph_construction="union"):
        """Build the adaptive-bandwidth RBF affinity kernel ``M`` from ``ad.obsm``.

        Uses exact kNN, an adaptive Gaussian width (distance to the ``k//2``-th neighbor),
        a symmetric neighbor graph, and evaluates the kernel only on graph edges
        (``O(nnz * d)`` rather than the old dense ``O(n^2 * d)`` per-row loop).

        :param n_neighbors: neighbors for the graph (defaults to ``self.n_neighbors``).
        :param graph_construction: ``'union'`` or ``'intersection'`` symmetrization.
        """
        xp, sp = self.xp, self.sp
        k = n_neighbors if n_neighbors is not None else self.n_neighbors
        n = self.n_cells

        if self.verbose:
            print(f"Building kernel on {self.build_kernel_on} (exact kNN, k={k}) ...")

        X = np.asarray(self.ad.obsm[self.build_kernel_on], dtype=self.dtype)
        Xb = xp.asarray(X)
        dist, idx = self._knn(X, k)

        # adaptive bandwidth: distance to the (k//2)-th nearest neighbor
        sigma = dist[:, k // 2]

        # binary kNN adjacency (each cell -> its k neighbors, self included)
        rows = xp.repeat(xp.arange(n), k)
        cols = idx.ravel()
        ones = xp.ones(n * k, dtype=self.dtype)
        G = sp.csr_matrix((ones, (rows, cols)), shape=(n, n))

        if graph_construction == "union":
            sym = (G + G.T).astype(bool).astype(self.dtype)
        elif graph_construction in ("intersect", "intersection"):
            Gb = G.astype(bool).astype(self.dtype)
            sym = Gb.multiply(Gb.T)
        else:
            raise ValueError(
                f"graph_construction = {graph_construction} is not valid; use 'union' or 'intersection'."
            )

        # evaluate the RBF kernel only on the edges of the symmetric graph
        sym = sym.tocoo()
        r, c = sym.row, sym.col
        diff = Xb[r] - Xb[c]
        sq = (diff * diff).sum(axis=1)
        denom = sigma[r] * sigma[c]
        vals = xp.exp(-sq / denom)

        M = sp.csr_matrix((vals, (r, c)), shape=(n, n))
        self.kernel_matrix = M
        self.K = (M @ M.T).tocsr()
        self._Mnorm2 = float(self._to_host(M.multiply(M).sum()))
        if self.verbose:
            print(f"Kernel M: {n}x{n}, nnz={int(M.nnz)}; K nnz={int(self.K.nnz)}")

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def initialize_archetypes(self):
        """Select initial archetype cell indices via waypoint + greedy selection."""
        k = self.k
        if self.waypoint_proportion > 0:
            waypoint_ix = self._get_waypoint_centers(k)
            waypoint_ix = np.random.choice(
                waypoint_ix,
                int(len(waypoint_ix) * self.waypoint_proportion),
                replace=False,
            )
            from_greedy = self.k - len(waypoint_ix)
            if self.verbose:
                print(f"Selecting {len(waypoint_ix)} cells from waypoint initialization.")
        else:
            from_greedy = self.k

        greedy_ix = self._get_greedy_centers(n_mcs=from_greedy + 10)
        if self.verbose:
            print(f"Selecting {from_greedy} cells from greedy initialization.")

        if self.waypoint_proportion > 0:
            all_ix = np.hstack([waypoint_ix, greedy_ix])
        else:
            all_ix = np.hstack([greedy_ix])

        unique_ix, ind = np.unique(all_ix, return_index=True)
        all_ix = unique_ix[np.argsort(ind)][:k]
        self.archetypes = all_ix

    def _get_waypoint_centers(self, n_waypoints=None):
        """Waypoint (max-min) sampling on diffusion components (Palantir).

        The diffusion-map eigendecomposition is the dominant cost of this one-time init
        step. When ``use_gpu`` is set and the installed Palantir exposes the ``use_gpu``
        option (with cupy available), that eigendecomposition runs on the GPU while the
        kNN kernel stays on the host, so results match the CPU path to ~1e-7. Otherwise
        it transparently falls back to the CPU solver.
        """
        import inspect

        import palantir

        k = n_waypoints if n_waypoints is not None else self.k
        ad = self.ad
        pca_components = pd.DataFrame(ad.obsm[self.build_kernel_on]).set_index(ad.obs_names)

        dm_kwargs = {}
        if self.use_gpu:
            try:
                sig = inspect.signature(palantir.utils.run_diffusion_maps)
                if "use_gpu" in sig.parameters:
                    from palantir._gpu import is_gpu_available

                    if is_gpu_available():
                        dm_kwargs["use_gpu"] = True
            except Exception:
                pass

        if self.verbose:
            where = "GPU eig" if dm_kwargs.get("use_gpu") else "CPU"
            print(f"Computing diffusion components from {self.build_kernel_on} for waypoints ({where}) ...")
        dm_res = palantir.utils.run_diffusion_maps(
            pca_components, n_components=self.n_neighbors, **dm_kwargs
        )
        dc_components = palantir.utils.determine_multiscale_space(dm_res, n_eigs=self.n_waypoint_eigs)

        if self.verbose:
            print("Sampling waypoints ...")
        waypoint_init = palantir.core._max_min_sampling(data=dc_components, num_waypoints=k)
        dc_components["iix"] = np.arange(len(dc_components))
        waypoint_ix = dc_components.loc[waypoint_init]["iix"].values
        return waypoint_ix

    def _get_greedy_centers(self, n_mcs=None):
        """Greedy adaptive column subset selection (CSSP) on the Gram matrix ``K``.

        Runs on the active backend. The inner projection is vectorized (two matmuls)
        instead of the old Python ``for r in range(j)`` loop.
        """
        xp = self.xp
        K = self.K
        n = self.n_cells
        k = n_mcs if n_mcs is not None else self.k

        if self.verbose:
            print("Initializing residual matrix using greedy column selection")

        f = xp.asarray(K.multiply(K).sum(axis=0)).ravel()
        g = xp.asarray(K.diagonal()).ravel()

        omega = xp.zeros((k, n), dtype=f.dtype)
        centers = np.zeros(k, dtype=int)

        for j in tqdm(range(k), disable=not self.verbose):
            score = f / g
            p = int(self._to_host(xp.argmax(score)))

            # p-th column of K (K is symmetric) via one-hot matvec (backend-agnostic)
            ep = xp.zeros(n, dtype=K.dtype)
            ep[p] = 1
            delta_term1 = K.dot(ep).ravel()

            # projection onto previously selected directions (vectorized)
            if j > 0:
                omega_j = omega[:j]  # (j, n)
                delta_term2 = omega_j.T.dot(omega_j[:, p])
            else:
                delta_term2 = xp.zeros(n, dtype=f.dtype)
            delta = delta_term1 - delta_term2

            delta_p = delta[p]
            delta_p = delta_p if delta_p > 0 else xp.asarray(0.0, dtype=delta.dtype)
            o = delta / xp.maximum(xp.sqrt(delta_p), 1e-6)

            omega_square_norm = xp.linalg.norm(o) ** 2
            omega_hadamard = o * o
            term1 = omega_square_norm * omega_hadamard

            if j > 0:
                omega_j = omega[:j]
                pl = omega_j.T.dot(omega_j.dot(o))
            else:
                pl = xp.zeros(n, dtype=f.dtype)
            ATAo = K.dot(o).ravel()
            term2 = o * (ATAo - pl)

            f = f - 2.0 * term2 + term1
            g = g + omega_hadamard
            omega[j, :] = o
            centers[j] = p

        return centers

    def initialize(self, initial_archetypes=None, initial_assignments=None):
        """Initialize ``B`` (archetypes) and ``A`` (assignments) given the kernel."""
        if self.K is None:
            raise RuntimeError("Must first construct kernel matrix before initializing SEACells.")
        xp = self.xp
        n = self.n_cells

        if initial_archetypes is not None:
            if self.verbose:
                print("Using provided list of initial archetypes")
            self.archetypes = np.asarray(initial_archetypes)

        if self.archetypes is None:
            self.initialize_archetypes()

        self.k = len(self.archetypes)
        k = self.k

        # B0: one-hot columns at archetype cells
        B0 = xp.zeros((n, k), dtype=self.xp.float64 if not self.use_gpu else self.xp.float32)
        arch = xp.asarray(self.archetypes)
        B0[arch, xp.arange(k)] = 1.0
        self.B0 = B0
        B = B0.copy()

        if initial_assignments is not None:
            A0 = xp.asarray(initial_assignments)
            assert A0.shape == (k, n), f"Initial assignment matrix should be of shape (k={k} x n={n})"
        else:
            A0 = xp.asarray(np.random.random((k, n)))
            A0 /= A0.sum(0)
            if self.verbose:
                print("Randomly initialized A matrix.")

        self.A0 = A0
        A = self._updateA(B, A0.copy())

        self.A_ = A
        self.B_ = B

        RSS = self.compute_RSS(A, B)
        self.RSS_iters.append(RSS)
        if self.convergence_threshold is None:
            self.convergence_threshold = self.convergence_epsilon * RSS
            if self.verbose:
                print(f"Setting convergence threshold at {self.convergence_threshold:.5f}")

    # ------------------------------------------------------------------ #
    # Frank-Wolfe updates
    # ------------------------------------------------------------------ #
    def _updateA(self, B, A_prev):
        """Frank-Wolfe update of the assignment matrix ``A`` (k x n) given ``B``.

        The FW step ``A <- A + f (e - A) = (1-f) A + f e`` (with ``e`` a one-hot column
        selector) is applied in place via a scatter-add, avoiding materializing the dense
        ``k x n`` selector every inner iteration.
        """
        xp = self.xp
        n, k = B.shape
        A = A_prev

        t2 = (self.K.dot(B)).T  # (k, n)
        t1 = t2.dot(B)  # (k, k)

        cols = xp.arange(n)
        t = 0
        while t < self.max_FW_iter:
            G = 2.0 * (t1.dot(A) - t2) - self.l2_penalty * A
            amins = xp.argmin(G, axis=0)
            f = 2.0 / (t + 2.0)
            A = (1.0 - f) * A
            A[amins, cols] += f
            t += 1
        return A

    def _updateB(self, A, B_prev):
        """Frank-Wolfe update of the archetype matrix ``B`` (n x k) given ``A``.

        Dense update: recomputes ``K @ B`` each inner iteration. The scatter step
        ``B <- (1-f) B + f e`` is applied in place via a scatter-add.
        """
        xp = self.xp
        k, n = A.shape
        B = B_prev
        t1 = A.dot(A.T)
        t2 = self.K.dot(A.T)
        cols = xp.arange(k)
        t = 0
        while t < self.max_FW_iter:
            G = 2.0 * (self.K.dot(B).dot(t1) - t2)
            amins = xp.argmin(G, axis=0)
            f = 2.0 / (t + 2.0)
            B = (1.0 - f) * B
            B[amins, cols] += f
            t += 1
        return B

    # ------------------------------------------------------------------ #
    # Objective
    # ------------------------------------------------------------------ #
    def compute_RSS(self, A=None, B=None):
        """Reconstruction error ``||M - MBA||_F`` in memory-scalable reduced form.

        Uses ``||M - MBA||^2 = ||M||^2 - 2 tr(KBA) + tr(B^T K B  A A^T)`` (with
        ``K = M^T M`` and symmetric ``M``), which needs only ``O(n*s)`` memory instead
        of forming the ``n x n`` reconstruction.
        """
        xp = self.xp
        if A is None:
            A = self.A_
        if B is None:
            B = self.B_
        if A is None or B is None:
            raise RuntimeError("Either assignment matrix A or archetype matrix B is None.")

        KB = self.K.dot(B)  # (n, k)
        cross = float(self._to_host((KB * A.T).sum()))
        C = B.T.dot(KB)  # (k, k) = B^T K B
        D = A.dot(A.T)  # (k, k) = A A^T
        quad = float(self._to_host((C * D).sum()))
        val = self._Mnorm2 - 2.0 * cross + quad
        return float(np.sqrt(max(val, 0.0)))

    def compute_reconstruction(self, A=None, B=None):
        """Return the (dense) reconstruction ``M B A``. Warning: ``n x n``; use for small data."""
        if A is None:
            A = self.A_
        if B is None:
            B = self.B_
        if A is None or B is None:
            raise RuntimeError("Either assignment matrix A or archetype matrix B is None.")
        return (self.kernel_matrix.dot(B)).dot(A)

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def step(self):
        """One alternating-minimization iteration (update A then B)."""
        if self.K is None:
            raise RuntimeError("Kernel matrix has not been computed. Run construct_kernel_matrix() first.")
        if self.A_ is None or self.B_ is None:
            raise RuntimeError("Model not initialized. Run initialize() first.")

        A = self._updateA(self.B_, self.A_)
        B = self._updateB(A, self.B_)
        self.RSS_iters.append(self.compute_RSS(A, B))
        self.A_ = A
        self.B_ = B

        labels = self.get_hard_assignments()
        self.ad.obs["SEACell"] = labels["SEACell"]

    def _fit(self, max_iter=50, min_iter=10, initial_archetypes=None, initial_assignments=None):
        self.initialize(initial_archetypes=initial_archetypes, initial_assignments=initial_assignments)

        converged = False
        n_iter = 0
        while (not converged and n_iter < max_iter) or n_iter < min_iter:
            n_iter += 1
            if self.verbose and (n_iter == 1 or n_iter % 10 == 0):
                print(f"Starting iteration {n_iter}.")
            self.step()
            if self.verbose and (n_iter == 1 or n_iter % 10 == 0):
                print(f"Completed iteration {n_iter}.")
            if np.abs(self.RSS_iters[-2] - self.RSS_iters[-1]) < self.convergence_threshold:
                if self.verbose:
                    print(f"Converged after {n_iter} iterations.")
                converged = True

        self.Z_ = self.B_.T @ self.K
        labels = self.get_hard_assignments()
        self.ad.obs["SEACell"] = labels["SEACell"]
        if not converged:
            raise RuntimeWarning(
                "Warning: Algorithm has not converged - you may need to increase the maximum number of iterations"
            )

    def fit(self, max_iter=100, min_iter=10, initial_archetypes=None, initial_assignments=None):
        """Fit the model (alternating Frank-Wolfe until convergence)."""
        if max_iter < min_iter:
            raise ValueError("max_iter is lower than min_iter.")
        self._fit(
            max_iter=max_iter,
            min_iter=min_iter,
            initial_archetypes=initial_archetypes,
            initial_assignments=initial_assignments,
        )

    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #
    def get_archetype_matrix(self):
        """Return the archetype matrix ``Z = B^T K`` (as a host array)."""
        return self._to_host(self.Z_)

    def get_hard_assignments(self):
        """Return a DataFrame assigning each cell to its argmax SEACell."""
        amax = self._to_host(self.A_.argmax(0)).astype(int)
        df = pd.DataFrame({"SEACell": [f"SEACell-{i}" for i in amax]})
        df.index = self.ad.obs_names
        df.index.name = "index"
        return df

    def get_hard_archetypes(self):
        """Return the names of the cells most strongly identified as archetypes."""
        return self.ad.obs_names[self._to_host(self.B_.argmax(0))]

    def get_soft_assignments(self):
        """Return top-5 soft SEACell labels and weights per cell."""
        archetype_labels = self.get_hard_archetypes()
        A = np.array(self._to_host(self.A_).T, copy=True)

        labels, weights = [], []
        for _ in range(5):
            l = A.argmax(1)
            labels.append(archetype_labels[l])
            weights.append(A[np.arange(A.shape[0]), l])
            A[np.arange(A.shape[0]), l] = -1

        weights = np.vstack(weights).T
        labels = np.vstack(labels).T
        soft_labels = pd.DataFrame(labels)
        soft_labels.index = self.ad.obs_names
        return soft_labels, weights

    def plot_convergence(self, save_as=None, show=True):
        """Plot RSS over iterations."""
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self.RSS_iters)
        plt.title("Reconstruction Error over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Squared Error")
        if save_as is not None:
            plt.savefig(save_as, dpi=150)
        if show:
            plt.show()
        plt.close()

    def save_assignments(self, outdir):
        """Save kernel, A, B (as scipy sparse ``.npz``) and hard assignments (csv)."""
        import os

        from scipy.sparse import csr_matrix, save_npz

        os.makedirs(outdir, exist_ok=True)
        M = csr_matrix(self._to_host(self.kernel_matrix))
        A = csr_matrix(self._to_host(self.A_)).T
        B = csr_matrix(self._to_host(self.B_))
        save_npz(outdir + "/kernel_matrix.npz", M)
        save_npz(outdir + "/A.npz", A)
        save_npz(outdir + "/B.npz", B)
        self.get_hard_assignments().to_csv(outdir + "/SEACells.csv")
