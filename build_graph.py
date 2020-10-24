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

##########################################################
# Projection matrix for error approximation
##########################################################

def get_projection(mat, cols):
    """
    mat is n x n
    cols n x d, project onto H

    both inputs should be sparse matrices
    also returns a csr matrix (supposedly, at least)

    result will be n x n"""
    print("Computing denominator...")
    denom = csr_matrix(np.linalg.pinv(cols.T.dot(cols).toarray()))
    print("Computing full matrix...")
    return (cols @ denom) @ (cols.T @ mat)

def projection_w(mat, cols):
    """Return n x d matrix, giving fractional membership"""
    print("Computing denominator...")
    denom = csr_matrix(np.linalg.pinv(cols.T.dot(cols).toarray()))
    print("Computing full matrix...")
    return (denom @ (cols.T @ mat)).T

##########################################################
# Helper functions for clustering
##########################################################

def get_new_center_for_cluster(embedding, assignments, cluster_idx):
    """Update the cluster center to medoid
    """
    cluster_members = np.arange(assignments.shape[0])[assignments == cluster_idx]
    cluster_coordinates = embedding[cluster_members, :]

    # get pairwise distances for points in this cluster
    distance_mtx = cdist(embedding, cluster_coordinates, metric="euclidean")

    # medoid: point with minimum maximum distance to other points
    medoid_idx = np.argmin(np.sum(distance_mtx, axis=1))

    # return index of the new cluster center
    return medoid_idx

##########################################################
# Helper functions for parallelizing kernel construction
##########################################################

def logsumexp_row_nonzeros(X):
    """Sparse logsumexp"""
    result = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        result[i] = logsumexp(X.data[X.indptr[i]:X.indptr[i+1]])
    return result

def normalize_row_nonzeros(X, logsum):
    """Sparse logsumexp"""
    for i in range(X.shape[0]):
        X.data[X.indptr[i]:X.indptr[i+1]] = np.exp(X.data[X.indptr[i]:X.indptr[i+1]] - logsum[i])
    return X

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

def rbf_for_row(G, data, median_distances, i):

    # convert row to binary numpy array
    row_as_array = G[i,:].toarray().ravel()

    # compute distances ||x - y||^2
    numerator = np.sum(np.square(data[i,:] - data), axis=1, keepdims=False)

    # compute radii
    denominator = median_distances[i] * median_distances

    # exp
    full_row = np.exp(-numerator / denominator)

    # masked row
    masked_row = np.multiply(full_row, row_as_array)

    return lil_matrix(masked_row)

def jaccard_for_row(G, row_sums, i):
    intersection = G[i,:].dot(G.T)
    subset_sizes = row_sums[i] + row_sums
    return lil_matrix(intersection / (subset_sizes.reshape(1,-1) - intersection))

##########################################################
# AVS + Markov
##########################################################

class MetacellGraph:

    """ Model that uses adaptive volume sampling to select representative cell types
    Then assigns cells to representative clusters using Markov random walk absorption probabilities

    Attributes:
        n (int): number of samples
        d (int): dimensionality of input (usually # PCA components)
        Y (numpy ndarray): input data (PCA components, shape n x d)
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

    def rbf(self, k:int=15):
        """Initialize adaptive bandwith RBF kernel (as described in C-isomap)

        Inputs:
            k (int): number of nearest neighbors for RBF kernel
        """

        if self.verbose:
            print("Computing kNN graph...")

        # compute kNN and the distance from each point to its nearest neighbors
        knn_graph = kneighbors_graph(self.Y, k, mode="connectivity", include_self=True)
        knn_graph_distances = kneighbors_graph(self.Y, k, mode="distance", include_self=True)

        if self.verbose:
            print("Computing radius for adaptive bandwidth kernel...")

        # compute median distance for each point amongst k-nearest neighbors
        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            median = k // 2
            median_distances = parallel(delayed(kth_neighbor_distance)(knn_graph_distances, median, i) for i in tqdm(range(self.n)))

        # convert to numpy array
        median_distances = np.array(median_distances)

        # take AND

        if self.verbose:
            print("Making graph symmetric...")
        sym_graph = (knn_graph + knn_graph.T > 0).astype(float)

        if self.verbose:
            print("Computing RBF kernel...")

        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            similarity_matrix_rows = parallel(delayed(rbf_for_row)(sym_graph, self.Y, median_distances, i) for i in tqdm(range(self.n)))

        if self.verbose:
            print("Building similarity LIL matrix...")

        similarity_matrix = lil_matrix((self.n, self.n))
        for i in tqdm(range(self.n)):
            similarity_matrix[i] = similarity_matrix_rows[i]

        if self.verbose:
            print("Constructing CSR matrix...")

        self.M = (similarity_matrix).tocsr()
        return self.M @ self.M.T

    def jaccard(self, k:int):
        """Uses Jaccard similarity between nearest neighbor sets as PSD kernel"""
        if self.verbose:
            print("Computing kNN graph...")

        knn_graph = kneighbors_graph(self.Y, k, include_self=True)

        # take AND
        if self.verbose:
            print("Making graph symmetric...")
        sym_graph = ((knn_graph.multiply(knn_graph.T)) > 0).astype(float)
        row_sums = np.sum(sym_graph, axis=1)

        if self.verbose:
            print("Computing Jaccard similarity...")

        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            similarity_matrix_rows = parallel(delayed(jaccard_for_row)(sym_graph, row_sums, i) for i in tqdm(range(self.n)))

        if self.verbose:
            print("Building similarity LIL matrix...")

        similarity_matrix = lil_matrix((self.n, self.n))
        for i in tqdm(range(self.n)):
            similarity_matrix[i] = similarity_matrix_rows[i]

        if self.verbose:
            print("Constructing CSR matrix...")

        self.M = similarity_matrix.tocsr()
        self.G = (self.M > 0).astype(float)

    def compute_transition_probabilities(self):
        """Normalize similarity matrix so that it represents transition probabilities"""

        # check to make sure there's a G and M
        if self.G is None or self.M is None:
            print("Need to initialize kernel first")
            return

        # compute row sums
        logM = self.G.copy()
        logM.data = np.log(logM.data)

        # compute sum in log space
        log_row_sums = logsumexp_row_nonzeros(logM)

        # normalize rows using logprobs and log sums
        logM = normalize_row_nonzeros(logM, log_row_sums)

        # save probabilities
        self.T = logM

    def compute_diffusion_map(self, k:int, t:int):
        """ diffusion embedding
        k: number of components for SVD
        t: number of time steps for random walk"""
        if self.T is None:
            print("Need to initialize transition matrix first!")
            return

        if self.verbose:
            print("Computing eigendecomposition")

        # right eigenvectors
        w, v = eigs(self.T, k=k, which="LM")
        #u, s, v = svds(self.T, k=k)

        order = np.argsort(-(np.real(w)))

        w = w[order]
        v = v[:,order]

        # set embedding
        lamb = np.power(np.real(w), t)
        self.embedding = np.real(v) @ np.diag(lamb)

        # store eigenvalues and eigenvectors
        self.eigenvalues = w
