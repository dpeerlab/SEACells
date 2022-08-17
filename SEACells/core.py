import numpy as np
import pandas as pd
import palantir
from tqdm.notebook import tqdm
import copy

from . import build_graph


class SEACells:
    """
    Fast kernel archetypal analysis.
    Finds archetypes and weights given annotated data matrix.
    Modifies annotated data matrix in place to include SEACell assignments in ad.obs['SEACell']
    """

    def __init__(self,
                 ad,
                 build_kernel_on: str,
                 n_SEACells: int,
                 verbose: bool = True,
                 n_waypoint_eigs: int = 10,
                 n_neighbors: int = 15,
                 convergence_epsilon: float=1e-3,
                 l2_penalty: float =0,
                 max_franke_wolfe_iters: int=50):
        """

        """
        self.ad = ad
        self.build_kernel_on = build_kernel_on
        self.n_cells = ad.shape[0]

        if not isinstance(n_SEACells, int):
            try:
                n_SEACells = int(n_SEACells)
            except:
                raise ValueError(f'The number of SEACells specified must be an integer type, not {type(n_SEACells)}')

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
        self.B0 = None

        return

    def add_precomputed_kernel_matrix(self, K):
        """

        """

        assert K.shape == (self.n_cells,
                          self.n_cells), f'Dimension of kernel matrix must be n_cells = ({self.n_cells},{self.n_cells}), not {K.shape} '
        self.kernel_matrix = K

        # Pre-compute dot product
        self.K = self.kernel_matrix @ self.kernel_matrix.T

    def construct_kernel_matrix(self, n_neighbors: int = None):
        """

        """
        # input to graph construction is PCA/SVD
        kernel_model = build_graph.SEACellGraph(self.ad, self.build_kernel_on, verbose=self.verbose)

        # K is a sparse matrix representing input to SEACell alg
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        M = kernel_model.rbf(n_neighbors)
        self.kernel_matrix = M

        # Pre-compute dot product
        self.K = self.kernel_matrix @ self.kernel_matrix.T

        return

    def initialize_archetypes(self):
        """
        Initialize B matrix which defines cells as SEACells. Selects waypoint_proportion from waypoint analysis,
        and the remainder by greedy selection.
        
        Modifies self.archetypes in-place with the indices of cells that are used as initialization for archetypes
        """
        k = self.k

        if self.waypoint_proportion > 0:
            waypoint_ix = self._get_waypoint_centers(k)
            waypoint_ix = np.random.choice(waypoint_ix, int(len(waypoint_ix) * self.waypoint_proportion), replace=False)
            from_greedy = self.k - len(waypoint_ix)
            if self.verbose:
                print(f'Selecting {len(waypoint_ix)} cells from waypoint initialization.')

        else:
            from_greedy = self.k

        greedy_ix = self._get_greedy_centers(n_mcs=from_greedy + 10)
        if self.verbose:
            print(f'Selecting {from_greedy} cells from greedy initialization.')

        if self.waypoint_proportion > 0:
            all_ix = np.hstack([waypoint_ix, greedy_ix])
        else:
            all_ix = np.hstack([greedy_ix])

        unique_ix, ind = np.unique(all_ix, return_index=True)
        all_ix = unique_ix[np.argsort(ind)][:k]
        self.archetypes = all_ix

    def initialize(self, initial_archetypes=None, initial_assignments=None):
        """

        """
        K = self.K
        # initialize B (update this to allow initialization from RRQR)
        n = K.shape[0]
        k = self.k

        if initial_archetypes is not None:
            if self.verbose:
                print('Using provided list of initial archetypes')
            self.archetypes = initial_archetypes

        if self.archetypes is None:
            self.initialize_archetypes()

        # Construction of B matrix
        B0 = np.zeros((n, k))
        all_ix = self.archetypes
        idx1 = list(zip(all_ix, np.arange(k)))
        B0[tuple(zip(*idx1))] = 1
        self.B0 = B0
        B = self.B0

        if initial_assignments is not None:
            raise NotImplementedError('Direct initialization of assignments of cells to SEACells not yet implemented.')
        else:
            A = np.random.random((k, n))
            A /= A.sum(0)
            A = self._updateA(B, A)

            if self.verbose:
                print('Randomly initialized A matrix.')

        self.A_ = A
        self.B_ = B
        # Create convergence threshold
        RSS = self.compute_RSS(A, B)
        self.RSS_iters.append(RSS)

        if self.convergence_threshold is None:
            self.convergence_threshold = self.convergence_epsilon * RSS
            if self.verbose:
                print(f'Setting convergence threshold at {self.convergence_threshold:.5f}')

    def _get_waypoint_centers(self, n_waypoints=None):
        """
        Initialize B matrix using waypoint analysis, as described in Palantir.
        From https://www.nature.com/articles/s41587-019-0068-4

        :param n_waypoints: (int) number of SEACells to initialize using waypoint analysis. If None specified,
                        all SEACells initialized using this method.
        :return: B - (array) n_datapoints x n_SEACells matrix with initial SEACell definitions
        """

        if n_waypoints == None:
            k = self.k
        else:
            k = n_waypoints

        ad = self.ad

        if self.build_kernel_on == 'X_pca':
            pca_components = pd.DataFrame(ad.obsm['X_pca']).set_index(ad.obs_names)
        elif self.build_kernel_on == 'X_svd':
            # Compute PCA components from ad object
            pca_components = pd.DataFrame(ad.obsm['X_svd']).set_index(ad.obs_names)
        else:
            pca_components = pd.DataFrame(ad.obsm[self.build_kernel_on]).set_index(ad.obs_names)

        print(f'Building kernel on {self.build_kernel_on}')

        if self.verbose:
            print(f'Computing diffusion components from {self.build_kernel_on} for waypoint initialization ... ')

        dm_res = palantir.utils.run_diffusion_maps(pca_components, n_components=self.n_neighbors)
        dc_components = palantir.utils.determine_multiscale_space(dm_res, n_eigs=self.n_waypoint_eigs)
        if self.verbose:
            print('Done.')

        # Initialize SEACells via waypoint sampling
        if self.verbose:
            print('Sampling waypoints ...')
        waypoint_init = palantir.core._max_min_sampling(data=dc_components, num_waypoints=k)
        dc_components['iix'] = np.arange(len(dc_components))
        waypoint_ix = dc_components.loc[waypoint_init]['iix'].values
        if self.verbose:
            print('Done.')

        return waypoint_ix

    def _get_greedy_centers(self, n_mcs=None):
        """Initialize SEACells using fast greedy adaptive CSSP

        From https://arxiv.org/pdf/1312.6838.pdf
        :param n_mcs: (int) number of SEACells to initialize using greedy selection. If None specified,
                        all SEACells initialized using this method.
        :return: B - (array) n_datapoints x n_SEACells matrix with initial SEACell definitions
        """

        K = self.K
        n = K.shape[0]

        if n_mcs is None:
            k = self.k
        else:
            k = n_mcs

        if self.verbose:
            print("Initializing residual matrix using greedy column selection")

        # precompute M.T * M
        # ATA = M.T @ M
        ATA = K

        if self.verbose:
            print("Initializing f and g...")

        f = np.array((ATA.multiply(ATA)).sum(axis=0)).ravel()
        # f = np.array((ATA * ATA).sum(axis=0)).ravel()
        g = np.array(ATA.diagonal()).ravel()

        d = np.zeros((k, n))
        omega = np.zeros((k, n))

        # keep track of selected indices
        centers = np.zeros(k, dtype=int)

        # sampling
        for j in tqdm(range(k)):

            score = f / g
            p = np.argmax(score)

            # print residuals
            residual = np.sum(f)

            delta_term1 = ATA[:, p].toarray().squeeze()
            # print(delta_term1)
            delta_term2 = np.multiply(omega[:, p].reshape(-1, 1), omega).sum(axis=0).squeeze()
            delta = delta_term1 - delta_term2

            # some weird rounding errors
            delta[p] = np.max([0, delta[p]])

            o = delta / np.max([np.sqrt(delta[p]), 1e-6])
            omega_square_norm = np.linalg.norm(o) ** 2
            omega_hadamard = np.multiply(o, o)
            term1 = omega_square_norm * omega_hadamard

            # update f (term2)
            pl = np.zeros(n)
            for r in range(j):
                omega_r = omega[r, :]
                pl += np.dot(omega_r, o) * omega_r

            ATAo = (ATA @ o.reshape(-1, 1)).ravel()
            term2 = np.multiply(o, ATAo - pl)

            # update f
            f += -2. * term2 + term1

            # update g
            g += omega_hadamard

            # store omega and delta
            d[j, :] = delta
            omega[j, :] = o

            # add index
            centers[j] = int(p)

        return centers

    def _updateA(self, B, A_prev):
        """
        Given archetype matrix B and using kernel matrix K, compute assignment matrix A using gradient descent.

        :param B: (array) n*k matrix (dense) defining SEACells as weighted combinations of cells
        :return: A: (array) k*n matrix (dense) defining weights used for assigning cells to SEACells
        """
        n, k = B.shape

        A = A_prev

        t = 0  # current iteration (determine multiplicative update)

        # precompute some gradient terms
        t2 = (self.K @ B).T
        t1 = t2 @ B

        # update rows of A for given number of iterations
        while t < self.max_FW_iter:
            # compute gradient (must convert matrix to ndarray)
            G = 2. * np.array(t1 @ A - t2) - self.l2_penalty*A

            # get argmins
            amins = np.argmin(G, axis=0)

            # loop free implementation
            e = np.zeros((k, n))
            e[amins, np.arange(n)] = 1.

            A += 2. / (t + 2.) * (e - A)
            t += 1

        return A

    def _updateB(self, A, B_prev):
        """
        Given assignment matrix A and using kernel matrix K, compute archetype matrix B

        :param A: (array) k*n matrix (dense) defining weights used for assigning cells to SEACells
        :return: B: (array) n*k matrix (dense) defining SEACells as weighted combinations of cells
        """

        K = self.K
        k, n = A.shape

        B = B_prev

        # keep track of error
        t = 0

        # precompute some terms
        t1 = A @ A.T
        t2 = K @ A.T

        # update rows of B for a given number of iterations
        while t < self.max_FW_iter:
            # compute gradient (need to convert np.matrix to np.array)
            G = 2. * np.array(K @ B @ t1 - t2)

            # get all argmins
            amins = np.argmin(G, axis=0)

            e = np.zeros((n, k))
            e[amins, np.arange(k)] = 1.

            B += 2. / (t + 2.) * (e - B)

            t += 1

        return B

    def compute_reconstruction(self, A=None, B=None):
        """
        Compute reconstructed data matrix using learned archetypes (SEACells) and assignments

        :param A: (array) k*n matrix (dense) defining weights used for assigning cells to SEACells
                If None provided, self.A is used.
        :param B: (array) n*k matrix (dense) defining SEACells as weighted combinations of cells
                If None provided, self.B is used.
        :return: array (n data points x data dimension) representing reconstruction of original data matrix
        """
        if A is None:
            A = self.A_
        if B is None:
            B = self.B_

        if A is None or B is None:
            raise RuntimeError('Either assignment matrix A or archetype matrix B is None.')
        return (self.kernel_matrix.dot(B)).dot(A)

    def compute_RSS(self, A=None, B=None):
        """
        Compute residual sum of squares error in difference between reconstruction and true data matrix
        :param A: (array) k*n matrix (dense) defining weights used for assigning cells to SEACells
                If None provided, self.A is used.
        :param B: (array) n*k matrix (dense) defining SEACells as weighted combinations of cells
        :param B: (array) n*k matrix (dense) defining SEACells as weighted combinations of cells
                If None provided, self.B is used.
        :return:
            ||X-XBA||^2 - (float) square difference between true data and reconstruction
        """
        if A is None:
            A = self.A_
        if B is None:
            B = self.B_

        reconstruction = self.compute_reconstruction(A, B)
        return np.linalg.norm(self.kernel_matrix - reconstruction)


    def plot_convergence(self, save_as=None, show=True):
        """
        Plot behaviour of squared error over iterations.
        :param save_as: (str) name of file which figure is saved as. If None, no plot is saved.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure()
        plt.plot(self.RSS_iters)
        plt.title('Reconstruction Error over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel("Squared Error")
        if save_as is not None:
            plt.savefig(save_as, dpi=150)
        if show:
            plt.show()
        plt.close()


    def step(self):
        """
        Perform one iteration of fitting to update A and B assignment matrices.
        """
        A = self.A_
        B = self.B_

        if self.K is None:
            raise RuntimeError(
                'Kernel matrix has not been computed. Run model.construct_kernel_matrix() first.')

        if A is None:
            raise RuntimeError('Cell to SEACell assignment matrix has not been initialised. Run model.initialize() first.')

        if B is None:
            raise RuntimeError('Archetype matrix has not been initialised. Run model.initialize() first.')

        A = self._updateA(B, A)
        B = self._updateB(A, B)

        self.RSS_iters.append(self.compute_RSS(A, B))

        self.A_ = A
        self.B_ = B

        # Label cells by SEACells assignment
        labels = self.get_hard_assignments()
        self.ad.obs['SEACell'] = labels['SEACell']

        return

    def _fit(self, max_iter: int = 50, min_iter:int=10, initial_archetypes=None, initial_assignments=None):
        """
        Compute archetypes and loadings given kernel matrix K. Iteratively updates A and B matrices until maximum
        number of iterations or convergence has been achieved.

        Modifies ad.obs in place to add 'SEACell' labels to cells.

        :param max_iter: (int) maximum number of iterations to update A and B matrices
        :param min_iter: (int) minimum number of iterations to update A and B matrices
        :param initial_archetypes: (list) indices of cells to use as initial archetypes

        """
        self.initialize(initial_archetypes=initial_archetypes, initial_assignments=initial_assignments)

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
            if np.abs(self.RSS_iters[-2] - self.RSS_iters[-1]) < self.convergence_threshold:
                if self.verbose:
                    print(f'Converged after {n_iter} iterations.')
                converged = True

        self.Z_ = self.B_.T @ self.K

        # Label cells by SEACells assignment
        labels = self.get_hard_assignments()
        self.ad.obs['SEACell'] = labels['SEACell']

        if not converged:
            raise RuntimeWarning(f'Warning: Algorithm has not converged - you may need to increase the maximum number of iterations')
        return

    def fit(self, max_iter: int = 100, min_iter:int=10, initial_archetypes=None):
        """
        Wrapper to fit model.

        :param max_iter: (int) maximum number of iterations to update A and B matrices. Default: 100
        :param min_iter: (int) maximum number of iterations to update A and B matrices. Default: 10
        :param initial_archetypes: (list) indices of cells to use as initial archetypes
        """
        if max_iter < min_iter:
            raise ValueError("The maximum number of iterations specified is lower than the minimum number of iterations specified.")
        self._fit(max_iter=max_iter, min_iter=min_iter, initial_archetypes=initial_archetypes, initial_assignments=None)

    def get_archetype_matrix(self):
        """Return k x n matrix of archetypes"""
        return self.Z_

    def get_soft_assignments(self):

        archetype_labels = self.get_hard_archetypes()
        A = copy.deepcopy(self.A_.T)

        labels = []
        weights = []
        for i in range(5):
            l = A.argmax(1)
            labels.append(archetype_labels[l])
            weights.append(A[np.arange(A.shape[0]), l])
            A[np.arange(A.shape[0]), l] = -1

        weights = np.vstack(weights).T
        labels = np.vstack(labels).T

        soft_labels = pd.DataFrame(labels)
        soft_labels.index = self.ad.obs_names

        return soft_labels, weights

    def get_hard_assignments(self):
        """
        Returns a dataframe with SEACell assignments under the column 'SEACell'
        :return: pd.DataFrame with column 'SEACell'
        """

        # Use argmax to get the index with the highest assignment weight
        bin_A = self.binarize_matrix_rows(self.A_)
        bin_B = self.binarize_matrix_rows(self.B_)

        labels = np.dot(bin_A.T, np.arange(bin_A.shape[0]))

        df = pd.DataFrame({'SEACell': labels.astype(int), 'is_MC': bin_B.sum(1).astype(bool)})
        df.index = self.ad.obs_names
        df.index.name = 'index'
        di = df[df['is_MC'] == True]['SEACell'].reset_index().set_index('SEACell').to_dict()['index']

        df['SEACell'] = df['SEACell'].map(di)

        return pd.DataFrame(df['SEACell'])

    def get_archetypes(self):
        """

        """
        raise NotImplementedError

    def get_hard_archetypes(self):
        """
        Return the names of cells most strongly identified as archetypes.
        """

        return self.ad.obs_names[self.B_.argmax(0)]


    @staticmethod
    def binarize_matrix_rows(T):
        """
        Convert matrix to binary form where the largest value in each row is 1 and all other values are 0
        :param T: (array) of floats
        :return bin_T: (array) with same shape as T, contains zeros everywhere except largest value in each row is 1.
        """
        bin_T = np.zeros(T.shape)
        bin_T[np.argmax(T, axis=0), np.arange(T.shape[1])] = 1
        return bin_T.astype(int)



    def get_archetypes(self):
        """
        Returns a list of cell-names identified as archetypes themselves.
        """


def sparsify_assignments(A, thresh: float):
    """
    :param A - is a cell x SEACells assignment matrix
    :param thresh -
    """
    A = copy.deepcopy(A)
    A[A < thresh] = 0

    # Renormalize
    A = A / A.sum(1, keepdims=True)
    A.sum(1)

    return A


def sparsify_assignments(A, thresh, keep_above_percentile=95):
    """
    :param: A is a cell x SEACells assignment matrix
    """
    A = copy.deepcopy(A)
    A[A == 0] = np.nan
    mins = np.nanpercentile(A, keep_above_percentile, axis=1).reshape(-1, 1)
    A = np.nan_to_num(A)
    mins[mins>thresh] = thresh
    A[A < mins] = 0

    # Renormalize
    A = A / A.sum(1, keepdims=True)

    return A



def summarize_by_soft_SEACell(ad, A, celltype_label=None, summarize_layer='raw', minimum_weight: float=0.05):
    """
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
    from scipy.sparse import csr_matrix
    import scanpy as sc

    compute_seacell_celltypes = False
    if celltype_label is not None:
        if not (celltype_label in ad.obs.columns):
            raise ValueError(f'Celltype label {celltype_label} not present in ad.obs')
        compute_seacell_celltypes = True

    if summarize_layer == 'raw' and ad.raw != None:
        data = ad.raw.X
    else:
        data = ad.layers[summarize_layer]

    A = sparsify_assignments(A.T, thresh=minimum_weight)

    seacell_expressions = []
    seacell_celltypes = []
    seacell_purities = []
    for ix in tqdm(range(A.shape[1])):
        cell_weights = A[:, ix]
        # Construct the SEACell expression using the
        seacell_exp = data.multiply(cell_weights[:, np.newaxis]).toarray().sum(0) / cell_weights.sum()
        seacell_expressions.append(seacell_exp)

        if compute_seacell_celltypes:
            # Compute the consensus celltype and the celltype purity
            cell_weights = pd.DataFrame(cell_weights)
            cell_weights.index = ad.obs_names
            purity = cell_weights.join(ad.obs[celltype_label]).groupby(celltype_label).sum().sort_values(by=0,
                                                                                                         ascending=False)
            purity = purity / purity.sum()
            celltype = purity.iloc[0]
            seacell_celltypes.append(celltype.name)
            seacell_purities.append(celltype.values[0])

    seacell_expressions = csr_matrix(np.array(seacell_expressions))
    seacell_ad = sc.AnnData(seacell_expressions, dtype=seacell_expressions.dtype)
    seacell_ad.var_names = ad.var_names
    seacell_ad.obs['Pseudo-sizes'] = A.sum(0)
    if compute_seacell_celltypes:
        seacell_ad.obs['celltype'] = seacell_celltypes
        seacell_ad.obs['celltype_purity'] = seacell_purities
    seacell_ad.var_names = ad.var_names
    return seacell_ad


def summarize_by_SEACell(ad, SEACells_label='SEACell', summarize_layer='raw'):
    """
    Aggregates cells within each SEACell, summing over all raw data for all cells belonging to a SEACell.
    Data is unnormalized and raw aggregated counts are stored .layers['raw'].
    Attributes associated with variables (.var) are copied over, but relevant per SEACell attributes must be
    manually copied, since certain attributes may need to be summed, or averaged etc, depending on the attribute.
    The output of this function is an anndata object of shape n_metacells x original_data_dimension.
    :return: anndata.AnnData containing aggregated counts.

    """
    from scipy.sparse import csr_matrix
    import scanpy as sc

    # Set of metacells
    metacells = ad.obs[SEACells_label].unique()

    # Summary matrix
    summ_matrix = pd.DataFrame(0.0, index=metacells, columns=ad.var_names)

    for m in tqdm(summ_matrix.index):
        cells = ad.obs_names[ad.obs[SEACells_label] == m]
        if summarize_layer == 'raw' and ad.raw != None:
            summ_matrix.loc[m, :] = np.ravel(ad[cells, :].raw.X.sum(axis=0))
        else:
            summ_matrix.loc[m, :] = np.ravel(ad[cells, :].layers[summarize_layer].sum(axis=0))

    # Ann data
    # Counts
    meta_ad = sc.AnnData(csr_matrix(summ_matrix), dtype=csr_matrix(summ_matrix).dtype)
    meta_ad.obs_names, meta_ad.var_names = summ_matrix.index.astype(str), ad.var_names
    meta_ad.layers['raw'] = csr_matrix(summ_matrix)
    return meta_ad
