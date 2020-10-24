# Uses adaptive sampling to pick rows of input matrix.
# Then uses Markov random walk absorption probabilities to assign cells.

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, dok_matrix, lil_matrix, diags, eye, csc_matrix, kron, vstack
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.sparse.linalg import svds, eigs, eigsh, norm, spsolve
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import t, entropy, multinomial

# for parallelizing stuff
from multiprocessing import cpu_count, Pool
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

# optimization
import time

# get number of cores for multiprocessing
NUM_CORES = cpu_count()

def kth_neighbor_distance(distances, k, i):
    """Returns distance to kth nearest neighbor
    Distances: sparse CSR matrix
    k: kth nearest neighbor
    i: index of row"""

    # convert row to 1D array
    row_as_array = distances[i,:].toarray().ravel()

    # number of nonzero elements
    num_nonzero = np.sum(row_as_array > 0)

    # argsort
    kth_neighbor_idx = np.argsort(np.argsort(-row_as_array)) == num_nonzero - k
    return np.linalg.norm(row_as_array[kth_neighbor_idx])

##########################################################
# AVS + Markov
##########################################################

class diffkm:

    """ Model that uses adaptive volume sampling to select representative cell types
    Then assigns cells to representative clusters using Markov random walk absorption probabilities

    Attributes:
        n (int): number of samples
        d (int): dimensionality of input (usually # PCA components)
        Y (numpy ndarray): input data
        M (CSR. matrix): similarity matrix
        G (CSR matrix): (binary) graph of connectivities
        T (CSR matrix): Markov (row-normalized) transition matrix
        verbose (bool):
    """

    def __init__(self, Y, n_cores:int=-1, verbose:bool=False):
        """Initialize model parameters"""
        # data parameters
        self.n, self.d = Y.shape

        # indices of each point
        self.indices = np.array(range(self.n))

        # save data
        self.Y = Y

        # number of cores for parallelization
        if n_cores != -1:
            self.num_cores = n_cores
        else:
            self.num_cores = NUM_CORES

        self.M = None # similarity matrix
        self.G = None # graph
        self.T = None # transition matrix

        # model params
        self.verbose=verbose

    ##############################################################
    # Methods related to kernel + sim matrix construction
    ##############################################################

    def initialize_kernel_rbf_parallel(self, k:int):
        """Initialize adaptive bandwith RBF kernel (as described in C-isomap)
        
        Inputs:
            k (int): number of nearest neighbors for RBF kernel
        """

        if self.verbose:
            print("Computing kNN graph...")

        # compute kNN and the distance from each point to its nearest neighbors
        # normalize Y
        #Y_normed = self.Y / np.sum(self.Y, axis=1, keepdims=True)
        knn_graph = kneighbors_graph(self.Y, k, mode="connectivity", include_self=True)
        #knn_graph = kneighbors_graph(Y_normed, k, mode="connectivity", metric="manhattan", include_self=True)
        
        #knn_graph_distances = kneighbors_graph(Y_normed, k, mode="distance", metric="manhattan", include_self=True)
        knn_graph_distances = kneighbors_graph(self.Y, k, mode="distance", include_self=True)

        if self.verbose:
            print("Computing radius for adaptive bandwidth kernel...")

        # compute median distance for each point amongst k-nearest neighbors
        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            median = 2
            median_distances = parallel(delayed(kth_neighbor_distance)(knn_graph_distances, median, i) for i in tqdm(range(self.n)))

        # convert to numpy array
        median_distances = csr_matrix(np.array(median_distances).reshape(-1,1))

        # make kernel matrix
        km = knn_graph_distances.power(2).multiply(median_distances.power(-2))
        km.data = np.exp(-km.data)

        # # sym_graph = knn_graph

        # if self.verbose:
        #     print("Computing RBF kernel...")

        # with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
        #     similarity_matrix_rows = parallel(delayed(rbf_for_row)(sym_graph, self.Y, median_distances, i) for i in tqdm(range(self.n)))

        # if self.verbose:
        #     print("Building similarity LIL matrix...")

        # similarity_matrix = lil_matrix((self.n, self.n))
        # for i in tqdm(range(self.n)):
        #     similarity_matrix[i] = similarity_matrix_rows[i]

        # if self.verbose:
        #     print("Constructing CSR matrix...")

        # self.M = (similarity_matrix).tocsr()
        # self.G = (self.M > 0).astype(float)
        # return self.M
        self.M = km + eye(km.shape[0])
        self.G = (self.M > 0).astype(float)
        return self.M
