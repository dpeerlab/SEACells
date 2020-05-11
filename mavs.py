# Uses adaptive sampling to pick rows of input matrix.
# Then uses Markov random walk absorption probabilities to assign cells.

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, dok_matrix, lil_matrix, diags, eye
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import svds, eigs, eigsh, norm, spsolve
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

# for parallelizing stuff
from multiprocessing import cpu_count, Pool
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

# optimization
import time

# get number of cores for multiprocessing
NUM_CORES = cpu_count()

##########################################################
# Helper functions for AVS
##########################################################

def projection_matrix(A):
    """Returns projection matrix P for subspace A (A is sparse)"""
    if np.ndim(A) == 1:
        A = A.reshape(1,-1)
    return csr_matrix(A @ np.linalg.inv((A.T @ A).toarray()) @ A.T)

def projection_matrix_dense(A):
    """Returns projection matrix P for subspace A (A is dense)"""
    if np.ndim(A) == 1:
        A = A.reshape(-1,1)
    try:
        return A @ np.linalg.inv(A.T @ A) @ A.T
    except:
        print(A)

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

class mavs:

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
        knn_graph = kneighbors_graph(self.Y, k, mode="connectivity", include_self=True)
        knn_graph_distances = kneighbors_graph(self.Y, k, mode="distance", include_self=True)

        if self.verbose:
            print("Computing radius for adaptive bandwidth kernel...")

        # compute median distance for each point amongst k-nearest neighbors
        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            median_distances = parallel(delayed(kth_neighbor_distance)(knn_graph_distances, k//2, i) for i in tqdm(range(self.n)))

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

        self.M = similarity_matrix.tocsr()
        self.G = (self.M > 0).astype(float)

    def initialize_kernel_jaccard_parallel(self, k:int):
        """Uses Jaccard similarity between nearest neighbor sets as PSD kernel"""
        if self.verbose:
            print("Computing kNN graph...")

        knn_graph = kneighbors_graph(self.Y, k, include_self=True)

        # take AND
        if self.verbose:
            print("Making graph symmetric...")
        sym_graph = ((knn_graph + knn_graph.T) > 0).astype(float)
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
        logM = self.M.copy()
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
        #self.embedding = v @ np.diag(s)

    ##############################################################
    # Clustering and sampling
    ##############################################################

    def adaptive_volume_sampling_original(self, k:int):
        """Adaptive volume sampling step

        Right now sampling the similarity matrix, but maybe we want to change this to something else.

        Inputs:
            k (int): how many centers do you want to deal with?
        """
        # keep a running list of selected centers
        S = set([])
        E = self.M.copy()
        
        for it in tqdm(range(k)):

            if self.verbose:
                print("Beginning iteration %d" % it)
                print("Computing row probabilities...")

            # compute probability of selecting each row
            row_probabilities_raw = norm(E, axis=1) / norm(E)

            if self.verbose:
                print("Sampling new index...")

            # sample new row index
            # row_idx = np.random.choice(range(self.n), 1, p=row_probabilities_norm)
            #print(row_probabilities_raw[:100])

            row_idx = np.argmax(row_probabilities_raw)

            if self.verbose:
                print("Selected index %d" % row_idx)

            # add selected index
            S.add(row_idx)

            if self.verbose:
                print("Computing projection matrix...")

            P = projection_matrix(E[row_idx,:].T)

            E = E - E @ P.T
            #print("Time to subtract projection: %d" % (end-start))
            #print(np.linalg.norm(E[row_idx,:]))
        self.centers = np.array(list(S))

    def adaptive_volume_sampling(self, k:int):
        """Fast greedy adaptive CSSP

        From https://arxiv.org/pdf/1312.6838.pdf
        """
        A = self.T.copy().T

        print("Initializing residual matrix...")

        # precomute ATA
        ATA = A.T.dot(A)

        # initialization
        f = np.power(norm(A.T @ A, axis=0), 2)
        g = np.power(norm(A, axis=0), 2)

        d = np.zeros((k, self.n))
        omega = np.zeros((k, self.n))

        # keep track of selected indices
        S = set([])

        # sampling
        for j in tqdm(range(k)):

            # select point
            score = f/g
            p = np.argmax(score)
            #print(score[p])

            # update delta
            #start = time.time()
            #delta_term1 = (A.T @ A[:,p]).toarray().ravel()
            delta_term1 = (ATA[:,p]).toarray().ravel()
            delta_term2 = np.multiply(omega[:,p].reshape(-1,1), omega).sum(axis=0)
            delta = delta_term1 - delta_term2
            #end = time.time()
            #print("Time to compute delta: %.4f" % (end-start))
            #print(delta)
            
            # update omega
            # print(delta_term1)
            # print(delta_term2)
            # print(delta[p])
            #start = time.time()
            o = delta / np.sqrt(delta[p])
            # update f (term1)
            omega_square_norm = np.linalg.norm(o)**2
            #print(omega_square_norm)
            omega_hadamard = np.multiply(o, o)
            term1 = omega_square_norm * omega_hadamard

            # update f (term2)
            pl = np.zeros(self.n)
            for r in range(j):
                omega_r = omega[r,:]
                pl += np.dot(omega_r, o) * omega_r
            #print(pl.shape)
            #end = time.time()
            #print("Time to compute term 1: %f" % (end-start))

            start = time.time()
            ATAo = (ATA @ o.reshape(-1,1)).ravel()
            #end = time.time()
            term2 = np.multiply(o, ATAo - pl)
            #end = time.time()
            #print("Time to compute term 2: %f" % (end-start))

            # update f
            f = (f - 2 * term2 + term1)
            #print(f[p])
            #print(f)

            # update g
            g = g #- np.multiply(o, o)
            #print(g[q])
            #print(g)

            # store omega and delta
            d[j,:] = delta
            omega[j,:] = o

            # add index
            S.add(p)

        self.centers = np.array(list(S))


    def adaptive_volume_sampling_slow(self, k:int, thres:float=1e-15):
        """Adaptive volume sampling step

        Right now sampling the similarity matrix, but maybe we want to change this to something else.

        Inputs:
            k (int): how many centers do you want to deal with?
        """
        # keep a running list of selected centers
        S = set([])
        # this doesn't get mutated. It's the original matrix that we want to apprximate
        # this is the residual matrix
        E = self.embedding.copy()
        
        for it in tqdm(range(k)):

            if np.linalg.norm(E) < thres:
                continue

            if self.verbose:
                print("Beginning iteration %d" % it)
                print("Computing row probabilities...")


            #row_probabilities_raw = np.linalg.norm(E, axis=1) / np.linalg.norm(E)
            #nums = np.zeros(self.n)
            #for i in range(self.n):
                #nums[i] = np.linalg.norm(E @ E[i,:].reshape(-1,1))
            nums = np.linalg.norm(E @ E.T, axis=0)
            denoms = np.linalg.norm(E, axis=1)
            scores = nums / denoms
            #row_probabilities_norm = row_probabilities_raw / sum(row_probabilities_raw)

            # sample new row index
            # row_idx = np.random.choice(range(self.n), 1, p=row_probabilities_norm)
            #print(row_probabilities_raw[:100])

            #row_idx = np.argmax(row_probabilities_raw)
            row_idx = np.argmax(scores)
            print(scores[row_idx])
            #print(row_probabilities_raw[row_idx])
            #print(row_idx)

            if self.verbose:
                print("Selected index %d" % row_idx)

            # add selected index
            S.add(row_idx)

            if self.verbose:
                print("Computing projection matrix...")

            # compute projection matrix for subspace
            #start = time.time()
            #print(E[row_idx,:])
            P = projection_matrix_dense(E[row_idx,:])
            #print(P.shape)
            #end = time.time()
            #print("Time to compute projection matrix: %d" % (end-start))

            # update E by subtracting its projection
            #print(E.shape)
            #print(P.shape)
            #E = E - P @ E

            E = E - E @ P.T
            #print("Time to subtract projection: %d" % (end-start))
            #print(np.linalg.norm(E[row_idx,:]))
        self.centers = np.array(list(S))

    def assign_clusters(self):
        """Assign clusters based on Markov absorption probabilities

        """
        # transition matrix for nonabsorbing states
        nonabsorbing_states = np.array([idx not in self.centers for idx in self.indices])
        Q = self.T[:,nonabsorbing_states][nonabsorbing_states,:]

        # compute fundamental matrix 
        F_inverse = eye(sum(nonabsorbing_states)) - Q

        # get matrix of nonabsorbing states
        R = self.T[nonabsorbing_states,:][:,self.centers]

        # compute absorption probabilities with spsolve
        B = spsolve(F_inverse, R)

        assignments_nonabsorbing = (B == B.max(axis=1)).astype(int)

        # compute boolean assignments
        self.assignments_bool = np.zeros((self.n, len(self.centers)))
        self.assignments_bool[nonabsorbing_states,:] = assignments_nonabsorbing
        self.assignments_bool[self.centers,:] = np.eye(len(self.centers))

        # compute absorption probabilities for each point (soft assignment)
        self.abs_probs = np.zeros((self.n, len(self.centers)))
        self.abs_probs[nonabsorbing_states,:] = B / B.sum(axis=1)
        self.abs_probs[self.centers,:] = np.eye(len(self.centers))

        # square
        self.W = self.abs_probs

    def cluster(self, k:int):
        """Wrapper for running adaptive volume sampling, then assigning each cluster.
        """
        self.adaptive_volume_sampling(k)
        self.assign_clusters_approximate()

    def cluster_original(self, k:int):
        """Wrapper for running adaptive volume sampling, then assigning each cluster.
        """
        self.adaptive_volume_sampling_original(k)
        self.assign_clusters()

    ##############################################################
    # Utils
    ##############################################################

    def get_metacell_sizes(self):
        """Returns array that gives the number of cells assigned to each metacell"""
        return self.assignments_bool.sum(axis=0)

    def get_soft_metacell_sizes(self):
        """Returns array that gives the number of cells assigned to each metacell"""
        return self.W.sum(axis=0)

    def _compute_modularity_matrix(self):
        """Computes modularity matrix B to give an idea of quality of clustering
        
        B_{ij} = A_{ij} - (k_i k_j)/2m
        where A is the adjacency matrix
        and k_i is the number of edges connected to node i
        """
        # get (try weighted adjacency matrix - want to penalize more similar cells from being assigned to diff clusters)
        A = self.M.copy()

        # get k
        k = (A).sum(axis=1)

        # get m (number of edges in the graph)
        m = np.sum(k)

        # this matrix might get really big because it won't be sparse...
        # B = A - k @ k.T / (2.*m)
        return A, k

    def compute_modularity(self):
        """Computes the modularity score given the current cluster assignments

        Use matrix formation: https://en.wikipedia.org/wiki/Modularity_(networks)
        """
        A, k = self._compute_modularity_matrix()

        # get m, number of edges in the graph
        m = np.sum(k)

        # get boolean assignments matrix
        # S = self.assignments_bool
        S = self.assignments_bool

        return 1./(2*m) * np.trace(S.T @ A @ S - S.T @ k @ k.T @ S / (2*m))

    def get_metacell_coordinates(self, coordinates=None, exponent:float=2.):
        if coordinates is None:
            coordinates = self.Y
        W = np.power(self.W, exponent)
        W = W / W.sum(axis=0, keepdims=True)
        return W.T @ coordinates

    ##############################################################
    # Analysis
    ##############################################################

    def get_cluster_mean_and_cov(self, coordinates, cluster_idx):
        """Given coordinates computes the gene expression mat
        Returns tuple of mean, cov
        """
        # unit normalize
        normalized_coordinates = coordinates / coordinates.sum(axis=1).reshape(-1,1)
        #print(normalized_coordinates.shape)

        # get weights
        weights = (self.W[:,cluster_idx] / np.sum(self.W[:,cluster_idx])).reshape(-1,1)
        #print(weights.shape)

        # mean
        cluster_mean = weights.T @ normalized_coordinates
        #print(cluster_mean)

        # get covariance
        #cluster_cov = np.multiply(weights.T, (normalized_coordinates - cluster_mean).T) @ (normalized_coordinates - cluster_mean)
        cluster_cov = np.cov(normalized_coordinates.T, aweights=weights.ravel())
        #print(cluster_cov)
        return cluster_mean, cluster_cov

    def get_expected_mean_and_cov(self, coordinates, cluster_idx):
        """Basically the same as above, but computes expected covariance given the cluster mean"""
        mean, cov = self.get_cluster_mean_and_cov(coordinates, cluster_idx)
        expected_cov = - mean.T @ mean
        mean_squeeze = np.array(mean).squeeze(0)
        expected_cov = expected_cov - np.diag(np.diag(expected_cov)) + np.diag(mean_squeeze * (1-mean_squeeze))
        return mean, expected_cov