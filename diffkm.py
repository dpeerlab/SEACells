# Uses adaptive sampling to pick rows of input matrix.
# Then uses Markov random walk absorption probabilities to assign cells.

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, dok_matrix, lil_matrix, diags, eye, csc_matrix
from sklearn.neighbors import kneighbors_graph
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
# Helper functions for clustering
##########################################################

def get_new_center_for_cluster(embedding, assignments):
        """Update the cluster center to medoid
        """
        cluster_members = assignments == cluster_idx
        cluster_coordinates = embedding[cluster_members, :]

        # get pairwise distances for points in this cluster
        distance_mtx = cdist(cluster_coordinates, cluster_coordinates, metric="euclidean")

        # medoid: point with minimum maximum distance to other points
        medoid_idx = np.argmin(np.argmax(distance_mtx, axis=1))

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

    ##############################################################
    # Clustering and sampling
    ##############################################################

    def kmpp(self, k):
        """kmeans++ initialization for medoids

        k: number of centroids
        """
        # array for storing centroids
        self.centers = np.zeros(k)

        # select initial point randomly
        new_point = np.random.choice(range(self.n), 1)
        self.centers[0] = new_point

        # initialize min distances
        distances = cdist(self.embedding, self.embedding[new_point, :], metric="euclidean").ravel()

        # assign rest of points
        for ix in range(1, k):
            new_point = np.argmax(distances)
            self.centers[ix] = new_point

            # get distance from all poitns to new points
            new_point_distances = cdist(self.embedding, self.embedding[new_point, :], metric="euclidean").ravel()

            # update min distances
            combined_distances = np.vstack([distances, new_point_distances])
            distances = np.argmin(combined_distances, axis=0)

    def assign_hard_clusters(self, k):
        """Use k-medoids to assign hard cluster labels"""
        distances = cdist(self.embedding, self.embedding[self.centers,:], metric="euclidean")
        self.assignments = np.argmin(distances, axis=1)

    def get_new_centers(self):
        """Wrapper for updating all cluster centers in parallel"""
        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            new_centers = parallel(delayed(get_new_center_for_cluster)(self.embedding, self.assignments) for i in tqdm(range(len(self.centers))))
        
        return new_centers

    def assign_soft_clusters(self):
        """Assign clusters based on Markov absorption probabilities

        """
        print("Assigning clusters...")
        # transition matrix for nonabsorbing states
        nonabsorbing_states = np.array([idx not in self.centers for idx in self.indices])
        Q = self.T[:,nonabsorbing_states][nonabsorbing_states,:]

        # compute fundamental matrix 
        F_inverse = (eye(sum(nonabsorbing_states)) - Q).tocsc()

        # get matrix of nonabsorbing states
        R = self.T[nonabsorbing_states,:][:,self.centers].tocsc()

        # compute absorption probabilities with spsolve
        B = spsolve(F_inverse, R)

        assignments_nonabsorbing = np.zeros(B.shape)
        max_idx = np.array(np.argmax(B, axis=1)).ravel()
        row_idx = np.arange(B.shape[0])
        assignments_nonabsorbing[row_idx, max_idx] = 1

        # compute boolean assignments
        self.assignments_bool = np.zeros((self.n, len(self.centers)))
        self.assignments_bool[nonabsorbing_states,:] = assignments_nonabsorbing
        self.assignments_bool[self.centers,:] = np.eye(len(self.centers))

        # compute absorption probabilities for each point (soft assignment)
        self.abs_probs = np.zeros((self.n, len(self.centers)))
        self.abs_probs[nonabsorbing_states,:] = B / B.sum(axis=1)

        self.abs_probs[self.centers,:] = np.eye(len(self.centers))

        # get weights
        self.W = self.abs_probs

    def cluster(self, k:int):
        """Wrapper for running adaptive volume sampling, then assigning each cluster.
        """
        # initialize with km++
        self.kmpp(k)

        # assign clusters
        self.assign_hard_clusters()

        converged=False
        while not converged:
            # update centers
            new_centers = self.get_new_centers()

            # update assignments
            self.assign_hard_clusters()

            # check convergence (no centers updated --> objectve local max)
            if np.all(np.equal(self.centers, new_centers)):
                converged=True

        self.assign_soft_clusters()

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

    def get_metacell_coordinates(self, coordinates=None, exponent:float=1.):
        if coordinates is None:
            coordinates = self.Y
        W = np.power(self.W, exponent)
        W = W / W.sum(axis=0, keepdims=True)
        return W.T @ coordinates

    ##############################################################
    # Identifying bad clusters
    ##############################################################

    def cluster_correlation(self, coordinates, idx:int):
        """Given coordinates and cluster index, get correlation
        Correlation is weighted by fractional membership

        Returns:
            correlation (d x d)
        """
        # get dimensions
        n, d = coordinates.shape

        # get empirical mean and covariance
        m, cov = self.get_cluster_mean_and_cov(coordinates, idx)

        # divide by square root of diagonal entries
        diags = np.diag(cov).reshape(-1, 1)

        # sqrt_diags
        sqrt_diags = np.sqrt(diags @ diags.T)

        # compute correlation
        corr = np.divide(cov, sqrt_diags)
        return corr


    def f_test_for_cluster(self, coordinates, idx:int, thres:float=0.05):
        """F test for a specific cluster
        Gets expected covariance and empirical covariance, using a weighted sum

        Inputs:
            coordinates (size_cluster * d)
            idx (int): cluster index
            thres (float): p-value threshold

        Returns:
            (bool) indicates whether or not one of the covariances is below threshold
        """
        # get expected mean and covariance
        # exp_mean, exp_cov = self.get_expected_mean_and_cov(coordinates, idx)

        # get correlation coefficients
        corr = self.cluster_correlation(coordinates, idx)

        # get degrees of freedom
        df = self.get_soft_metacell_sizes()[idx]

        # compute p values
        num = corr * np.sqrt(df - 2)
        denom = np.sqrt(1 - corr + 1e-12) # add small jitter for now...
        t_statistic = np.divide(num, denom)

        # get pearson r
        one_minus_p = t.cdf(t_statistic, df)
        p = (1 - one_minus_p)

        # check if below  thres
        below_thres = p < thres

        # set all diagonal entries to True
        for i in range(p.shape[0]):
            below_thres[i,i] = False

        # count the number of violations
        prop = np.sum(below_thres) / (corr.shape[0]**2)

        return prop, p

    def f_test(self, coordinates, threshold:float=0.05, max_prop:float=0.1):
        """Identify clusters with variance not satisfying multinomial assumption with F-test

        Input
            coordinates: (n x d)
            threshold: p value threshold

        Returns: indices of bad clusters
        """
        # TODO: parallelize cluster checking
        props = []
        out = np.zeros(len(self.centers), dtype=bool)
        for cluster in tqdm(range(len(self.centers))):
            prop, p = self.f_test_for_cluster(coordinates, cluster, threshold)
            out[cluster] = prop > max_prop
            props.append(prop)

        return np.arange(len(self.centers))[out], props

    ##############################################################
    # Likelihoods and BIC
    ##############################################################

    def compute_cluster_likelihood(self, coordinates, cluster_idx):
        """Compute cluster likelihood under multinomial distribution"""

        # get library sizes
        lib_sizes = np.array(np.sum(coordinates, axis=1)).ravel()
        #print(lib_sizes.shape)

        # get mean and covariance
        mean, cov = self.get_cluster_mean_and_cov(coordinates, cluster_idx)
        mean = np.array(mean).ravel()
        #print(mean.shape)

        coords = coordinates.toarray()
        #print(coords.shape)

        # get  likelihood
        logp = multinomial.logpmf(coords, lib_sizes, mean)

        weighted_logp = np.multiply(self.W[:,cluster_idx], logp)
        #print(logp.shape)
        return np.sum(weighted_logp)

    def compute_total_likelihood(self, coordinates):
        """Compute total likelihood of data under multinomial sampling """

        logp = 0.
        for cluster_idx in tqdm(range(len(self.centers))):
            logp += self.compute_cluster_likelihood(coordinates, cluster_idx)

        return logp

    def bic(self, coordinates):
        """Compute Bayesian information criterion under multinomial model"""

        logp = self.compute_total_likelihood(coordinates)
        k = len(self.centers) * coordinates.shape[1]
        n = coordinates.shape[0]

        print("Number of parameters: ", k)
        print("Number of points: ", n)
        print("Log likelihood: ", logp)

        bic = k * np.log(n) - 2 * logp

        print("BIC: ", bic)
        return bic

    ##############################################################
    # Entropy
    ##############################################################

    def compute_cell_entropy(self):
        """Compute entropy of each cell for cluster assignment
        """
        return entropy(self.W.T)

    def compute_total_entropy(self):
        """Compute total entropy of all cells
        """
        return np.sum(entropy(self.W.T))

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