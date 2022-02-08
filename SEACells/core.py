import numpy as np
import pandas as pd
import palantir
from tqdm.notebook import tqdm

from . import build_graph


class SEACells:
    """
    Fast kernel archetypal analysis.
    Finds archetypes and weights given annotated data matrix.
    Modifies annotated data matrix in place to include SEACell assignments in ad.obs['SEACell']
    """

    def __init__(self,
                 ad,
                 build_kernel_on,
                 n_SEACells: int,
                 max_iter: int = 50,
                 verbose: bool = True,
                 true_B=None,
                 true_A=None,
                 n_waypoint_eigs: int = 10,
                 waypt_proportion: float = 1,
                 n_neighbors: int = 15,
                 convergence_epsilon=1e-5):
        """

        :param ad: AnnData object containing observations matrix to use for computing SEACells
        :param build_kernel_on: (str) key in ad.obsm which defines matrix to build SEACells on
        :param n_SEACells: (int) number of SEACells to use
        :param max_iter: (int) maximum number of iterations for gradient descent optimization of assignment matrices
        :param verbose: (bool) display additional information as program runs
        :param true_B: (array) true matrix defining cells as SEACells, if known
        :param true_A: (array) true matrix assigning cells to SEACells, if known
        :param n_waypoint_eigs: (int) number of eigenvalues to use when initializing SEACells from waypoint analysis
        :param waypt_proportion: (float) proportion of SEACells to initialize using waypoint method, remainder using greedy
        :param n_neighbors: (int) number of neighbors to use in building kNN graph
        :param convergence_epsilon: (float) stop optimizing when squared error is below this proportion of its original value
        """

        self.ad = ad
        if build_kernel_on not in ad.obsm:
            raise ValueError(f'Key {build_kernel_on} is not present in AnnData obsm.')

        self.build_kernel_on = build_kernel_on
        self.k = n_SEACells
        self.max_iter = max_iter
        self.verbose = verbose
        self.true_B = true_B
        self.true_A = true_A

        self.n_waypoint_eigs = n_waypoint_eigs
        self.waypoint_proportion = waypt_proportion

        self.n_neighbors = n_neighbors

        self.RSS_iters = []
        self.convergence_epsilon = convergence_epsilon
        self.convergence_threshold = None

        print

    def _initialize_archetypes(self):
        """
        Initialize B matrix which defines cells as SEACells. Selects waypoint_proportion from waypoint analysis,
        and the remainder by greedy selection.
        :return: B - (array) n_datapoints x n_SEACells matrix which initial SEACell definitions
        """
        K = self.K
        n = K.shape[0]
        k = self.k

        if self.waypoint_proportion > 0:
            waypt_ix = self._get_waypoint_centers(k)
            waypt_ix = np.random.choice(waypt_ix, int(len(waypt_ix) * self.waypoint_proportion))
            from_greedy = self.k - len(waypt_ix)
            if self.verbose:
                print(f'Selecting {len(waypt_ix)} cells from waypoint initialization.')

        else:
            from_greedy = self.k

        greedy_ix = self._get_greedy_centers(n_mcs=from_greedy + 10)
        if self.verbose:
            print(f'Selecting {from_greedy} cells from greedy initialization.')

        if self.waypoint_proportion > 0:
            all_ix = np.hstack([waypt_ix, greedy_ix])
        else:
            all_ix = np.hstack([greedy_ix])

        unique_ix, ind = np.unique(all_ix, return_index=True)
        all_ix = unique_ix[np.argsort(ind)][:k]

        B0 = np.zeros((n, k))
        idx1 = list(zip(all_ix, np.arange(k)))
        B0[tuple(zip(*idx1))] = 1

        return B0

    def _get_waypoint_centers(self, n_waypts=None):
        """
        Initialize B matrix using waypoint analysis, as described in Palantir.
        From https://www.nature.com/articles/s41587-019-0068-4

        :param n_waypts: (int) number of SEACells to initialize using waypoint analysis. If None specified,
                        all SEACells initialized using this method.
        :return: B - (array) n_datapoints x n_SEACells matrix with initial SEACell definitions
        """

        if n_waypts == None:
            k = self.k
        else:
            k = n_waypts

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

        dm_res = palantir.utils.run_diffusion_maps(pca_components)
        dc_components = palantir.utils.determine_multiscale_space(dm_res, n_eigs=self.n_waypoint_eigs)
        if self.verbose:
            print('Done.')

        # Initialize SEACells via waypoint sampling
        if self.verbose:
            print('Sampling waypoints ...')
        waypoint_init = palantir.core._max_min_sampling(data=dc_components, num_waypoints=k)
        dc_components['iix'] = np.arange(len(dc_components))
        waypt_ix = dc_components.loc[waypoint_init]['iix'].values
        if self.verbose:
            print('Done.')

        return waypt_ix

    def _get_greedy_centers(self, n_mcs=None):
        """Initialize SEACells using fast greedy adaptive CSSP

        From https://arxiv.org/pdf/1312.6838.pdf
        :param n_mcs: (int) number of SEACells to initialize using greedy selection. If None specified,
                        all SEACells initialized using this method.
        :return: B - (array) n_datapoints x n_SEACells matrix with initial SEACell definitions
        """

        K = self.K
        n = K.shape[0]

        if n_mcs == None:
            k = self.k
        else:
            k = n_mcs

        X = K

        if self.verbose:
            print("Initializing residual matrix using greedy column selection")

        # precompute A.T * A
        # ATA = K.T @ K
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
        while t < self.max_iter:
            # compute gradient (must convert matrix to ndarray)
            G = 2. * np.array(t1 @ A - t2)

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
        while t < self.max_iter:
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

        return np.dot(np.dot(self.ad.obsm[self.build_kernel_on].T, B), A)

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
        return np.linalg.norm(self.ad.obsm[self.build_kernel_on].T - reconstruction)

        return

    def plot_convergence(self, save_as=None):
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
        plt.show()
        plt.close()

    def _fit(self, max_iter: int = 50, min_iter:int=10, B0=None):
        """
        Compute archetypes and loadings given kernel matrix K. Iteratively updates A and B matrices until maximum
        number of iterations or convergence has been achieved.

        Modifies ad.obs in place to add 'SEACell' labels to cells.

        :param max_iter: (int) maximum number of iterations to update A and B matrices
        :param B0: (array) n_datapoints x n_SEACells initial guess of archetype matrix
        """

        if self.verbose:
            print('Building kernel...')

        # input to graph construction is PCA/SVD
        kernel_model = build_graph.SEACellGraph(self.ad, self.build_kernel_on, verbose=True)

        # K is a sparse matrix representing input to SEACell alg
        K = kernel_model.rbf(self.n_neighbors)
        self.K = K

        # initialize B (update this to allow initialization from RRQR)
        n = K.shape[0]
        k = self.k

        if self.true_B is None:
            if B0 is not None:
                if self.verbose:
                    print('Using provided initial B matrix')
                B = B0
                self.B0 = B0
            else:
                B = self._initialize_archetypes()
                self.B0 = B
        else:
            if self.verbose:
                print('Using fixed B matrix as provided.')
            B = self.true_B

        A = np.random.random((k, n))
        A /= A.sum(0)
        A = self._updateA(B, A)
        
        print('Randomly initialized A matrix.')

        # Create convergence threshold
        RSS = self.compute_RSS(A, B)
        self.RSS_iters.append(RSS)

        if self.convergence_threshold is None:
            self.convergence_threshold = self.convergence_epsilon * RSS
            if self.verbose:
                print(f'Setting convergence threshold at {self.convergence_threshold}')

        converged = False
        n_iter = 0
        while (not converged and n_iter < max_iter) or n_iter < min_iter:

            n_iter += 1

            if n_iter == 1 or (n_iter) % 10 == 0:
                print(f"Starting iteration {n_iter}.")
            if self.true_A is None:
                A = self._updateA(B, A)
            else:
                print('Not updating A, true A provided')
                A = self.true_A

            if self.true_B is None:
                B = self._updateB(A, B)
            else:
                print('Not updating B, true B provided')

            if n_iter == 1 or (n_iter) % 10 == 0:
                print(f"Completed iteration {n_iter}.")

            self.RSS_iters.append(self.compute_RSS(A, B))

            # Check for convergence
            if np.abs(self.RSS_iters[-2] - self.RSS_iters[-1]) < self.convergence_threshold:
                converged = True

            self.A_ = A
            self.B_ = B
            n_SEACells = self.get_assignments()['SEACell'].unique().shape[0]

        print(f'Converged after {n_iter} iterations.')
        self.A_ = A
        self.B_ = B
        self.Z_ = B.T @ self.K

        # Label SEACells as well as assignment entropy as proxy for SEACell 'confidence'
        labels = self.get_assignments()
        self.ad.obs['SEACell'] = labels['SEACell']


    def fit(self, n_iter: int = 8, waypoint_proportion: float = None, B0=None):
        """
        Wrapper to fit model given kernel matrix and max number of iterations

        :param n_iter: (int) maximum number of iterations to update A and B matrices
        :param waypoint_proportion: (float) proportion of SEACells to intialize using waypoint initializations
        :param B0: (array) n_datapoints x n_SEACells initial guess of archetype matrix
        """

        if waypoint_proportion is not None:
            self.waypoint_proportion = waypoint_proportion
        self._fit(n_iter, B0=B0)

    def get_archetypes(self):
        """Return k x n matrix of archetypes"""
        return self.Z_

    def get_centers(self):
        """Return closest point to each archetype"""
        return np.argmax(self.B_, axis=0)

    def get_soft_assignments(self):
        """Return archetype assignments for each point (n x k)
        """
        return self.A_.T

    def get_sizes(self):
        """Return size of each SEACell as array
        """
        return Counter(np.argmax(self.A_, axis=0))

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

    def get_assignments(self):
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
    meta_ad = sc.AnnData(csr_matrix(summ_matrix))
    meta_ad.obs_names, meta_ad.var_names = summ_matrix.index.astype(str), ad.var_names

    return meta_ad
