import cupy as cp
import cupyx
import numpy as np
import palantir
import pandas as pd
from sklearn.preprocessing import normalize
import scipy.sparse
from cupyx.scipy.sparse.linalg import norm
from tqdm import tqdm
import scipy
from icecream import ic

try:
    from . import build_graph
except ImportError:
    import build_graph


class SEACellsGPU:
    """GPU Implementation of SEACells algorithm.

    The implementation uses fast kernel archetypal analysis to find SEACells - groupings
    of cells that represent highly granular, distinct cell states. SEACells are found by solving a convex optimization
    problem that minimizes the residual sum of squares between the kernel matrix and the weighted sum of the archetypes.

    Modifies annotated data matrix in place to include SEACell assignments in ad.obs['SEACell']

    """

    def __init__(
        self,
        ad,
        build_kernel_on: str,
        n_SEACells: int,
        verbose: bool = True,
        n_waypoint_eigs: int = 10,
        n_neighbors: int = 15,
        convergence_epsilon: float = 1e-3,
        l2_penalty: float = 0,
        max_franke_wolfe_iters: int = 50,
    ):
        """GPU Implementation of SEACells algorithm.

        :param ad: (AnnData) annotated data matrix
        :param build_kernel_on: (str) key corresponding to matrix in ad.obsm which is used to compute kernel for metacells
                                Typically 'X_pca' for scRNA or 'X_svd' for scATAC
        :param n_SEACells: (int) number of SEACells to compute
        :param verbose: (bool) whether to suppress verbose program logging
        :param n_waypoint_eigs: (int) number of eigenvectors to use for waypoint initialization
        :param n_neighbors: (int) number of nearest neighbors to use for graph construction
        :param convergence_epsilon: (float) convergence threshold for Franke-Wolfe algorithm
        :param l2_penalty: (float) L2 penalty for Franke-Wolfe algorithm
        :param max_franke_wolfe_iters: (int) maximum number of iterations for Franke-Wolfe algorithm

        Class Attributes:
            ad: (AnnData) annotated data matrix
            build_kernel_on: (str) key corresponding to matrix in ad.obsm which is used to compute kernel for metacells
            n_cells: (int) number of cells in ad
            k: (int) number of SEACells to compute
            n_waypoint_eigs: (int) number of eigenvectors to use for waypoint initialization
            waypoint_proportion: (float) proportion of cells to use for waypoint initialization
            n_neighbors: (int) number of nearest neighbors to use for graph construction
            max_FW_iter: (int) maximum number of iterations for Franke-Wolfe algorithm
            verbose: (bool) whether to suppress verbose program logging
            l2_penalty: (float) L2 penalty for Franke-Wolfe algorithm
            RSS_iters: (list) list of residual sum of squares at each iteration of Franke-Wolfe algorithm
            convergence_epsilon: (float) algorithm converges when RSS < convergence_epsilon * RSS(0)
            convergence_threshold: (float) convergence threshold for Franke-Wolfe algorithm
            kernel_matrix: (csr_matrix) kernel matrix of shape (n_cells, n_cells)
            K: (csr_matrix) dot product of kernel matrix with itself, K = K @ K.T
            archetypes: (list) list of cell indices corresponding to archetypes
            A_: (csr_matrix) matrix of shape (k, n) containing final assignments of cells to SEACells
            B_: (csr_matrix) matrix of shape (n, k) containing archetype weights
            A0: (csr_matrix) matrix of shape (k, n) containing initial assignments of cells to SEACells
            B0: (csr_matrix) matrix of shape (n, k) containing initial archetype weights
        """
        print("Welcome to SEACells GPU!")
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

        self.n_waypoint_eigs = n_waypoint_eigs
        self.waypoint_proportion = 1
        self.n_neighbors = n_neighbors

        self.max_FW_iter = max_franke_wolfe_iters
        self.verbose = verbose
        self.l2_penalty = l2_penalty

        self.RSS_iters = []
        self.convergence_epsilon = convergence_epsilon
        self.convergence_threshold = None

        # Parameters to be initialized later in the model
        self.kernel_matrix = None
        self.K = None

        # Archetypes as list of cell indices
        self.archetypes = None

        self.A_ = None
        self.B_ = None
        self.A0 = None
        self.B0 = None

        # TODO: Remove this later -------
        # Create a new dataframe that will hold the sparsity ratios of A, B, K
        self.sparsity_ratios = pd.DataFrame(columns=["A", "B", "K"])

        return

    def add_precomputed_kernel_matrix(self, K):
        """Add precomputed kernel matrix to SEACells object.

        :param K: (np.ndarray) kernel matrix of shape (n_cells, n_cells)
        :return: None.
        """
        assert K.shape == (
            self.n_cells,
            self.n_cells,
        ), f"Dimension of kernel matrix must be n_cells = ({self.n_cells},{self.n_cells}), not {K.shape} "
        self.kernel_matrix = cupyx.scipy.sparse.csr_matrix(K)

        # Pre-compute dot product
        self.K = self.kernel_matrix.dot(self.kernel_matrix.transpose())

    def construct_kernel_matrix(
        self, n_neighbors: int = None, graph_construction="union"
    ):
        """Construct kernel matrix from data matrix using PCA/SVD and nearest neighbors.

        :param n_neighbors: (int) number of nearest neighbors to use for graph construction.
                            If none, use self.n_neighbors, which has a default value of 15.
        :param graph_construction: (str) method for graph construction. Options are 'union' or 'intersection'.
                                    Default is 'union', where the neighborhood graph is made symmetric by adding an edge
                                    (u,v) if either (u,v) or (v,u) is in the neighborhood graph. If 'intersection', the
                                    neighborhood graph is made symmetric by adding an edge (u,v) if both (u,v) and (v,u)
                                    are in the neighborhood graph.
        :return: None.
        """
        # TODO: make K sparse / check if already sparse

        # input to graph construction is PCA/SVD
        kernel_model = build_graph.SEACellGraph(
            self.ad, self.build_kernel_on, verbose=self.verbose
        )
        print("build_graph.SEACellGraph completed")

        # K is a sparse matrix representing input to SEACell alg
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        M = kernel_model.rbf(n_neighbors, graph_construction=graph_construction)
        self.kernel_matrix = cupyx.scipy.sparse.csr_matrix(M)

        # Pre-compute dot product
        self.K = self.kernel_matrix.dot(self.kernel_matrix.transpose())
        # ic(self.K.s)
        # ic(sle)
        # ic(type(self..))
        return

    def initialize_archetypes(self):
        """Initialize B matrix which defines cells as SEACells.

        Uses waypoint analysis for initialization into to fully cover the phenotype space, and then greedily
        selects the remaining cells (if redundant cells are selected by waypoint analysis).

        Modifies self.archetypes in-place with the indices of cells that are used as initialization for archetypes.

        By default, the proportion of cells selected by waypoint analysis is 1. This can be changed by setting the
        waypoint_proportion parameter in the SEACells object. For example, setting waypoint_proportion = 0.5 will
        select half of the cells by waypoint analysis and half by greedy selection.
        """
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
                print(
                    f"Selecting {len(waypoint_ix)} cells from waypoint initialization."
                )

        else:
            from_greedy = self.k

        greedy_ix = self._get_greedy_centers(n_SEACells=from_greedy + 10)
        # ic("_get_greedy_centers completed")

        if self.verbose:
            print(f"Selecting {from_greedy} cells from greedy initialization.")

        if self.waypoint_proportion > 0:
            # print("SELF.WAYPOINT_PROPORTION > 0")
            all_ix = np.hstack([waypoint_ix, greedy_ix])
        else:
            # print("SELF.WAYPOINT_PROPORTION <= 0")
            all_ix = np.hstack([greedy_ix])

        unique_ix, ind = np.unique(all_ix, return_index=True)

        all_ix = unique_ix[np.argsort(ind)][:k]
        self.archetypes = cp.array(all_ix)

    def initialize(self, initial_archetypes=None, initial_assignments=None):
        """Initialize the model.

        Initializes the B matrix (constructs archetypes from a convex combination of cells) and the A matrix
        (defines assignments of cells to archetypes).

        Assumes the kernel matrix has already been constructed. B matrix is of shape (n_cells, n_SEACells) and A matrix
        is of shape (n_SEACells, n_cells).

        :param initial_archetypes: (np.ndarray) initial archetypes to use for initialization. If None, use waypoint
                                     analysis and greedy selection to initialize archetypes.
        :param initial_assignments: (np.ndarray) initial assignments to use for initialization. If None, use
                                        random initialization.
        :return: None
        """
        if self.K is None:
            raise RuntimeError(
                "Must first construct kernel matrix before initializing SEACells."
            )
        # initialize B (update this to allow initialization from RRQR)
        n = self.K.shape[0]
        # print("n = self.K.shape[0]")
        # print(n)

        # print("initial_archetypes")
        # print(initial_archetypes)

        if initial_archetypes is not None:
            if self.verbose:
                print("Using provided list of initial archetypes")
            self.archetypes = initial_archetypes

        if self.archetypes is None:
            # print("STARTING INITIALIZE_ARCHETYPES")
            self.initialize_archetypes()
            # print("FINISHED INITIALIZE_ARCHETYPES")

        self.k = len(self.archetypes)
        k = self.k

        # print("k")
        # print(k)

        # Sparse construction of B matrix
        cols = cp.arange(k)
        # print("cols")
        # print(type(cols))
        rows = self.archetypes
        # print("rows")
        # print(type(rows))
        shape = (n, k)
        # print("shape")
        # print(shape)
        data = cp.ones(len(rows))
        # print(type(data))
        B0 = cupyx.scipy.sparse.csr_matrix(
            (cp.ones(len(rows)), (rows, cols)), shape=shape
        )
        # print("constructed b0")
        self.B0 = B0
        B = self.B0.copy()
        # print("constructed B0 and B")

        # print(initial_assignments)

        if initial_assignments is not None:
            A0 = initial_assignments
            assert A0.shape == (
                k,
                n,
            ), f"Initial assignment matrix should be of shape (k={k} x n={n})"
            A0 = cupyx.scipy.sparse.csr_matrix(A0)
            # Normalize axis 0 with l1 norm
            l1_norms  = norm(A0, ord=1, axis=0)
            l1_norms[l1_norms == 0] = 1.0
            A0 = A0.multiply(1.0 / l1_norms)

        else:
            # Need to ensure each cell is assigned to at least one archetype
            # Randomly sample roughly 25% of the values between 0 and k
            # ic("flag")
            archetypes_per_cell = int(k * 0.25)
            rows = np.random.randint(0, k, size=(n, archetypes_per_cell)).reshape(-1)
            columns = np.repeat(np.arange(n), archetypes_per_cell)
            # print(type(rows))
            # print(type(columns))

            # ic(rows.shape)
            # ic(columns.shape)
            # ic(cp.random.random(len(rows)).shape)
            # ic((k, n))

            # data = cp.random.random(len(rows))

            # Ensure that rows, columns, and data are 1D 
            # rows = rows.ravel()
            # columns = columns.ravel()
            # data = data.ravel()

            A0 = scipy.sparse.csr_matrix(
                (np.random.random(len(rows)), (rows, columns)), shape=(k, n)
            )
            # Normalize axis 0 with l1 norm
            A0 = cupyx.scipy.sparse.csc_matrix(normalize(A0, norm="l1", axis=0))

            if self.verbose:
                print("Randomly initialized A matrix.")

        self.A0 = A0
        A = self.A0.copy()

        # print("UPDATE A BEGINNING")
        # print(B.shape)
        # print(type(B))
        # print(A.shape)
        # print(type(A))
        A = self._updateA(B, A)
        # print("UPDATE A COMPLETED")

        self.A_ = A
        self.B_ = B

        # Create convergence threshold
        # print("RSS START")
        RSS = self.compute_RSS(A, B)
        # print("RSS FIN")
        self.RSS_iters.append(RSS)

        if self.convergence_threshold is None:
            self.convergence_threshold = self.convergence_epsilon * RSS
            # ic(type(self.convergence_threshold))
            # if self.verbose:
            #     print(
            #         f"Convergence threshold set to {self.convergence_threshold} based on epsilon = {self.convergence_epsilon}"
            #     )

    def _get_waypoint_centers(self, n_waypoints=None):
        """Initialize B matrix using waypoint analysis, as described in Palantir.

        From https://www.nature.com/articles/s41587-019-0068-4.

        :param n_waypoints: (int) number of SEACells to initialize using waypoint analysis. If None specified,
                        all SEACells initialized using this method.
        :return: (np.ndarray) indices of cells to use as initial archetypes
        """
        if n_waypoints is None:
            k = self.k
        else:
            k = n_waypoints

        ad = self.ad

        if self.build_kernel_on == "X_pca":
            pca_components = pd.DataFrame(ad.obsm["X_pca"]).set_index(ad.obs_names)
        elif self.build_kernel_on == "X_svd":
            # Compute PCA components from ad object
            pca_components = pd.DataFrame(ad.obsm["X_svd"]).set_index(ad.obs_names)
        else:
            pca_components = pd.DataFrame(ad.obsm[self.build_kernel_on]).set_index(
                ad.obs_names
            )

        print(f"Building kernel on {self.build_kernel_on}")

        if self.verbose:
            print(
                f"Computing diffusion components from {self.build_kernel_on} for waypoint initialization ... "
            )

        dm_res = palantir.utils.run_diffusion_maps(
            pca_components, n_components=self.n_neighbors
        )
        dc_components = palantir.utils.determine_multiscale_space(
            dm_res, n_eigs=self.n_waypoint_eigs
        )
        if self.verbose:
            print("Done.")

        # Initialize SEACells via waypoint sampling
        if self.verbose:
            print("Sampling waypoints ...")
        waypoint_init = palantir.core._max_min_sampling(
            data=dc_components, num_waypoints=k
        )
        dc_components["iix"] = np.arange(len(dc_components))
        waypoint_ix = dc_components.loc[waypoint_init]["iix"].values
        if self.verbose:
            print("Done.")

        return waypoint_ix

    def _get_greedy_centers(self, n_SEACells=None):
        """Initialize SEACells using fast greedy adaptive CSSP.

        From https://arxiv.org/pdf/1312.6838.pdf
        :param n_SEACells: (int) number of SEACells to initialize using greedy selection. If None specified,
                        all SEACells initialized using this method.
        :return: (np.ndarray) indices of cells to use as initial archetypes
        """
        n = self.K.shape[0]

        if n_SEACells is None:
            k = self.k
        else:
            k = n_SEACells

        if self.verbose:
            print("Initializing residual matrix using greedy column selection")

        # precompute M.T * M
        # ATA = M.T @ M
        ATA = self.K

        # Find out how many non-zero entries there are in ATA
        # ic(ATA.nnz)
        # ic(ATA.shape)
        # ic(ATA.nnz/ATA.shape[0]**2)

        if self.verbose:
            print("Initializing f and g...")

        f = cp.array((ATA.multiply(ATA)).sum(axis=0)).ravel()
        g = cp.array(ATA.diagonal()).ravel()

        d = cupyx.scipy.sparse.csr_matrix((k, n))
        omega = cupyx.scipy.sparse.csr_matrix((k, n))

        # keep track of selected indices
        centers = cp.zeros((k,), dtype=int)

        # sampling
        for j in tqdm(range(k)):
            # Compute score, dividing the sparse f by the sparse g
            score = f / g

            # Compute p, which is the largest score
            p = cp.argmax(score)

            # Compute delta_term1 to be the column of ATA at index p
            delta_term1 = ATA[:, p].toarray().squeeze()

            # Compute delta_term2 to be the sum of the outer product of omega and itself
            delta_term2 = (
                omega[:, p].reshape(-1, 1).multiply(omega).sum(axis=0).squeeze()
            )
            delta = delta_term1 - delta_term2

            # some weird rounding errors
            delta[p] = max([0, delta[p]])

            o = delta / max([cp.sqrt(delta[p]), 1e-6])
            omega_square_norm = cp.linalg.norm(o) ** 2
            omega_hadamard = cp.multiply(o, o)
            term1 = omega_square_norm * omega_hadamard

            # update f (term2)
            pl = cp.zeros(n)
            for r in range(j):
                omega_r = omega[r, :]
                pl += omega_r.dot(o) * omega_r

            ATAo = (ATA @ o.reshape(-1, 1)).ravel()
            term2 = o * (ATAo - pl)

            # update f
            f += -2.0 * term2 + term1

            # update g
            g += omega_hadamard

            # store omega and delta
            d[j, :] = delta
            omega[j, :] = o

            # add index
            centers[j] = int(p)

        return centers

    def _updateA(self, Bg, Ag):
        """Compute assignment matrix A using constrained gradient descent via Frank-Wolfe algorithm.

        Given archetype matrix B and using kernel matrix K, compute assignment matrix A using constrained gradient
        descent via Frank-Wolfe algorithm.

        :param B: (n x k csr_matrix) defining SEACells as weighted combinations of cells
        :param A_prev: (n x k csr_matrix) defining previous weights used for assigning cells to SEACells
        :return: (n x k csr_matrix) defining updated weights used for assigning cells to SEACells
        """
        # ic("updateA")
        n, k = Bg.shape

        t = 0  # current iteration (determine multiplicative update)
        # Kg = self.K.get() 
        Kg = self.K

        # Bg = Bg.get()

        # precompute some gradient terms
        t2g = (Kg.dot(Bg)).T
        t1g = t2g.dot(Bg)

        lambda_l1 = 0  
        lambda_l2 = 0

        # Make sure all the datatypes are float64
        # t1g = t1g.astype(cp.float64)
        # t2g = t2g.astype(cp.float64)
        # Kg = Kg.astype(cp.float64)
        # Bg = Bg.astype(cp.float64)
        # Ag = Ag.astype(cp.float64)

        # Ag = Ag.get()

        # update rows of A for given number of iterations
        while t < self.max_FW_iter:
            # # L1 regularization term: the weight times the L1 norm of the matrix
            # l1_term = lambda_l1 * norm(Ag, ord=1)
            # # L2 regularization term: 0.5 times the weight times the L2 norm of the matrix squared
            # l2_term = 0.5 * lambda_l2 * norm(Ag, ord = 'fro')**2

            # compute gradient
            Gg = 2.0 * (t1g @ Ag - t2g).get().toarray()
            amins = Gg.argmin(axis=0)
            amins = cp.array(amins).reshape(-1)
            # ic(amins.shape)
            # ic(type(amins))
            # loop free implementaton
            # eg = cp.zeros((k, n))
            # eg[amins, cp.arange(n)] = 1.0
            # eg = cupyx.scipy.sparse.csr_matrix(eg)

            # eg = cupyx.scipy.sparse.csr_matrix((cp.ones(len(amins)), (amins, cp.arange(n))), shape=Ag.shape)
            eg = cupyx.scipy.sparse.csr_matrix((cp.ones(len(amins)), (amins, cp.arange(n))), shape=Ag.shape)

            # row_indices = cp.array(amins)
            # col_indices = cp.arange(n)
            # data = cp.ones_like(row_indices, dtype = cp.float64)
            # eg = cupyx.scipy.sparse.coo_matrix((data, (row_indices, col_indices)), shape=(k, n))
            # eg = eg.tocsr()

            # print("eg")


            Ag += 2.0/(t+2.0) * (eg - Ag)
            # Ag = cp.add(Ag, cp.multiply(f, cp.subtract(eg, Ag)))
            t += 1
            # print("f, Ag, t")

        # A = Ag.get()
        # A = cupyx.scipy.sparse.csr_matrix(Ag)
        A = Ag

        del t1g, t2g, Kg, Gg, Bg, eg, amins, Ag
        cp._default_memory_pool.free_all_blocks()

        # Ag.data[Ag.data < 0.04] = 0

        # ic(type(A))

        return A

    def _updateB(self, Ag, Bg):
        """Compute archetype matrix B using constrained gradient descent via Frank-Wolfe algorithm.

        Given assignment matrix A and using kernel matrix K, compute archetype matrix B using constrained gradient
        descent via Frank-Wolfe algorithm.

        :param A: (n x k csr_matrix) defining weights used for assigning cells to SEACells
        :param B_prev: (n x k csr_matrix) defining previous SEACells as weighted combinations of cells
        :return: (n x k csr_matrix) defining updated SEACells as weighted combinations of cells
        """
        # ic("_updateB")
        k, n = Ag.shape

        # keep track of error
        t = 0

        Kg = self.K
        # precompute some terms
        t1g = Ag.dot(Ag.T)
        t2g = Kg.dot(Ag.T)

        # update rows of B for a given number of iterations
        while t < self.max_FW_iter:
            # compute gradient
            Gg = 2 * (Kg.dot(Bg).dot(t1g) - t2g)

            # get all argmins
            amins = Gg.argmin(axis=0)

            eg = cp.zeros((n, k))
            eg[amins, cp.arange(k)] = 1.0
            eg = cupyx.scipy.sparse.csr_matrix(eg)

            f = 2.0 / (t + 2.0)
            Bg = Bg + (f * (eg - Bg))

            t += 1

        del (
            t1g,
            t2g,
            Ag,
            Kg,
            Gg,
            eg,
            amins,
        )
        cp._default_memory_pool.free_all_blocks()

        return Bg

    def compute_reconstruction(self, A=None, B=None):
        """Compute reconstructed data matrix using learned archetypes (SEACells) and assignments.

        :param A: (k x n csr_matrix) defining weights used for assigning cells to SEACells
                If None provided, self.A is used.
        :param B: (n x k csr_matrix) defining SEACells as weighted combinations of cells
                If None provided, self.B is used.
        :return: (n x n csr_matrix) defining reconstructed data matrix.
        """
        if A is None:
            A = self.A_
        if B is None:
            B = self.B_

        if A is None or B is None:
            raise RuntimeError(
                "Either assignment matrix A or archetype matrix B is None."
            )

        # ic(type(A))
        # ic(type(B))
        # ic(type(self.kernel_matrix))

        # Print proportion of non-zero entries in A and B and self.kernel_matrix
        # ic(A.nnz / (A.shape[0]*A.shape[1]))
        # ic(B.nnz / (B.shape[0]*B.shape[1]))
        # ic(self.kernel_matrix.nnz / (self.kernel_matrix.shape[0]*self.kernel_matrix.shape[1]))

        # Add the sparsity ratios to the dataframe using pandas.concat
        self.sparsity_ratios = pd.concat(
            [
                self.sparsity_ratios,
                pd.DataFrame(
                    {
                        "A": A.nnz / (A.shape[0] * A.shape[1]),
                        "B": B.nnz / (B.shape[0] * B.shape[1]),
                        "K": self.kernel_matrix.nnz
                        / (self.kernel_matrix.shape[0] * self.kernel_matrix.shape[1]),
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

        # turn this to a dense calculation

        # turn A, B, and self.kernel_matrix to dense matrices
        A = A.todense()
        B = B.todense()
        K = self.kernel_matrix.todense()
        # A = A.get().todense()
        # B = B.get().todense()
        # K = self.kernel_matrix.get().todense()

        # K = self.kernel_matrix

        return (K.dot(B)).dot(A)

    def compute_RSS(self, A=None, B=None):
        """Compute residual sum of squares error in difference between reconstruction and true data matrix.

        :param A: (k x n csr_matrix) defining weights used for assigning cells to SEACells
                If None provided, self.A is used.
        :param B: (n x k csr_matrix) defining SEACells as weighted combinations of cells
                If None provided, self.B is used.
        :return:
            ||X-XBA||^2 - (float) square difference between true data and reconstruction.
        """
        if A is None:
            A = self.A_
        if B is None:
            B = self.B_

        # print("COMPUTE RECONSTRUCTION")
        reconstruction = self.compute_reconstruction(A, B)

        # Print proportion of nonzero values in reconstruction
        # ic(reconstruction.nnz / (reconstruction.shape[0]*reconstruction.shape[1]))
        # ic(type(reconstruction))

        # print("FINISHED COMPUTE RECONSTRUCTION")

        assert reconstruction.shape == self.kernel_matrix.shape
        assert reconstruction.shape == (
            self.n_cells,
            self.n_cells,
        ), "reconstruction.shape != (self.n_cells, self.n_cells)"

        # Densify the kernel matrix
        kernel_matrix = self.kernel_matrix.todense()
        # kernel_matrix = self.kernel_matrix
        diff = kernel_matrix - reconstruction
        # ic(type(diff))

        # Want to free up memory for A, B, and reconstruction
        del A, B, reconstruction
        cp._default_memory_pool.free_all_blocks()

        # convert diff to array
        # diff = diff.get()

        # ic(scipy.sparse.linalg.norm(self.kernel_matrix - reconstruction))
        # ic(cp.linalg.norm(diff))
        # ic(np.linalg.norm(diff))

        return np.linalg.norm(diff)

    def plot_convergence(self, save_as=None, show=True):
        """Plot behaviour of squared error over iterations.

        :param save_as: (str) name of file which figure is saved as. If None, no plot is saved.
        :param show: (bool) whether to show plot
        :return: None.
        """
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

    def step(self, iteration=None):
        """Perform one iteration of SEACell algorithm. Update assignment matrix A and archetype matrix B.

        If iterations = None, it checks if the RSS has converged every time. Else, it checks every 5th iteration.

        :return: None.
        """
        if iteration is None or (iteration is not None and iteration % 3 == 0):
            A = self.A_
            B = self.B_

            if self.K is None:
                raise RuntimeError(
                    "Kernel matrix has not been computed. Run model.construct_kernel_matrix() first."
                )

            if A is None:
                raise RuntimeError(
                    "Cell to SEACell assignment matrix has not been initialised. Run model.initialize() first."
                )

            if B is None:
                raise RuntimeError(
                    "Archetype matrix has not been initialised. Run model.initialize() first."
                )

            A = self._updateA(B, A)
            B = self._updateB(A, B)

            # ic(self.RSS_iters)
            RSS = self.compute_RSS(A, B)
            self.RSS_iters.append(RSS)
            # ic(self.RSS_iters)

            self.A_ = A
            self.B_ = B

            del A, B, RSS

            # Label cells by SEACells assignment
            labels = self.get_hard_assignments()
            self.ad.obs["SEACell"] = labels["SEACell"]

        return

    def _fit(
        self,
        max_iter: int = 50,
        min_iter: int = 10,
        initial_archetypes=None,
        initial_assignments=None,
    ):
        """Internal method to compute archetypes and loadings given kernel matrix K.

        Iteratively updates A and B matrices until maximum number of iterations or convergence has been achieved.

        Modifies ad.obs in place to add 'SEACell' labels to cells.
        :param max_iter: (int) maximum number of iterations to perform
        :param min_iter: (int) minimum number of iterations to perform
        :param initial_archetypes: (array) initial archetypes to use. If None, random initialisation is used.
        :param initial_assignments: (array) initial assignments to use. If None, random initialisation is used.
        :return: None
        """
        # print("RUNNING SELF.INITIALIZE")
        self.initialize(
            initial_archetypes=initial_archetypes,
            initial_assignments=initial_assignments,
        )
        # print("SELF.INITIALIZE COMPLETE")

        converged = False
        n_iter = 0
        while (not converged and n_iter < max_iter) or n_iter < min_iter:
            n_iter += 1
            if n_iter == 1 or (n_iter) % 10 == 0:
                if self.verbose:
                    print(f"Starting iteration {n_iter}.")
            self.step()

            if n_iter == 1 or (n_iter) % 10 == 0:
                if self.verbose:
                    print(f"Completed iteration {n_iter}.")

            # Check for convergence
            # if len(self.RSS_iters) > 1:
                # ic(self.RSS_iters[-2] - self.RSS_iters[-1])
                # ic(self.convergence_threshold)
            if (
                cp.abs(self.RSS_iters[-2] - self.RSS_iters[-1])
                < self.convergence_threshold
            ):
                if self.verbose:
                    print(f"Converged after {n_iter} iterations.")
                converged = True

        self.Z_ = self.B_.T @ self.K

        # Label cells by SEACells assignment
        labels = self.get_hard_assignments()
        self.ad.obs["SEACell"] = labels["SEACell"]

        if not converged:
            raise RuntimeWarning(
                "Warning: Algorithm has not converged - you may need to increase the maximum number of iterations"
            )
        return

    def fit(
        self,
        max_iter: int = 100,
        min_iter: int = 10,
        initial_archetypes=None,
        initial_assignments=None,
    ):
        """Compute archetypes and loadings given kernel matrix K.

        Iteratively updates A and B matrices until maximum number of iterations or convergence has been achieved.
        :param max_iter: (int) maximum number of iterations to perform (default 100)
        :param min_iter: (int) minimum number of iterations to perform (default 10)
        :param initial_archetypes: (array) initial archetypes to use. If None, random initialisation is used.
        :param initial_assignments: (array) initial assignments to use. If None, random initialisation is used.
        :return: None.
        """
        if max_iter < min_iter:
            raise ValueError(
                "The maximum number of iterations specified is lower than the minimum number of iterations specified."
            )
        self._fit(
            max_iter=max_iter,
            min_iter=min_iter,
            initial_archetypes=initial_archetypes,
            initial_assignments=initial_assignments,
        )

    def get_archetype_matrix(self):
        """Return k x n matrix of archetypes computed as the product of the archetype matrix B and the kernel matrix K."""
        return self.Z_

    def get_soft_assignments(self):
        """Return soft SEACells assignment.

        Returns a tuple of (labels, weights) where labels is a dataframe with SEACell assignments for the top 5
        SEACell assignments for each cell and weights is an array with the corresponding weights for each assignment.
        :return: (pd.DataFrame, np.array) with labels and weights.
        """
        import copy

        archetype_labels = self.get_hard_archetypes()
        A = copy.deepcopy(self.A_.T)

        labels = []
        weights = []
        for _i in range(5):
            l = A.argmax(1)
            labels.append(archetype_labels[l])
            weights.append(A[cp.arange(A.shape[0]), l])
            A[cp.arange(A.shape[0]), l] = -1

        weights = cp.vstack(weights).T
        labels = cp.vstack(labels).T

        soft_labels = pd.DataFrame(labels)
        soft_labels.index = self.ad.obs_names

        return soft_labels, weights

    def get_hard_assignments(self):
        """Return a dataframe with the SEACell assignment for each cell.

        The assignment is the SEACell with the highest assignment weight.

        :return: (pd.DataFrame) with SEACell assignments.
        """
        # Use argmax to get the index with the highest assignment weight

        # ic(self.A_.shape)
        # ic(self.A_.argmax(0).shape)
        # ic(self.A_.argmax(0).flatten().shape)

        # ic(self.ad.obs_names.shape)
        # ic("get_hard_assignments")
        df = pd.DataFrame(
            {"SEACell": [f"SEACell-{i}" for i in self.A_.argmax(axis=0).flatten()]}
        )
        df.index = self.ad.obs_names
        df.index.name = "index"

        return df

    def get_hard_archetypes(self):
        """Return the names of cells most strongly identified as archetypes.

        :return list of archetype names.
        """
        return self.ad.obs_names[self.B_.argmax(0)]

    def save_model(self, outdir):
        """Save the model to a pickle file.

        :param outdir: (str) path to directory to save to
        :return: None.
        """
        import pickle

        with open(outdir + "/model.pkl", "wb") as f:
            pickle.dump(self, f)
        return None

    def save_assignments(self, outdir):
        """Save SEACell assignments.

        Saves:
        (1) the cell to SEACell assignments to a csv file with the name 'SEACells.csv'.
        (2) the kernel matrix to a .npz file with the name 'kernel_matrix.npz'.
        (3) the archetype matrix to a .npz file with the name 'A.npz'.
        (4) the loading matrix to a .npz file with the name 'B.npz'.

        :param outdir: (str) path to directory to save to
        :return: None
        """
        import os

        os.makedirs(outdir, exist_ok=True)
        save_npz(outdir + "/kernel_matrix.npz", self.kernel_matrix)
        save_npz(outdir + "/A.npz", self.A_.T)
        save_npz(outdir + "/B.npz", self.B_)

        labels = self.get_hard_assignments()
        labels.to_csv(outdir + "/SEACells.csv")
        return None
