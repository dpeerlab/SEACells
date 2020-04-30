# An implementation of of the Partioning Around Medoids (PAM) algorithm on a graph
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, dok_matrix, lil_matrix, diags
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import svds, eigs, eigsh
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

# for parallelizing stuff
from multiprocessing import cpu_count, Pool
from joblib import Parallel, delayed
import tqdm

# get number of cores for multiprocessing
NUM_CORES = cpu_count()

#################################################
# Helper functions parallelization
#################################################

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

def update_centers_for_cluster(graph, cluster_assignments, cluster_idx):
    """Helper function of updating center of a given cluster
    Graph: connectivity graph. higher edge weight = higher distance
    """
    # all indices for points
    full_indices = np.array(range(graph.shape[0]))

    # indices of points assigned to cluster index
    indices = full_indices[cluster_assignments[:,cluster_idx].astype(bool)]

    # define subgraph
    subgraph = graph[indices,:][:,indices]

    # distance matrix defined by graph (try cosine distance, graph distance is slow)
    # dist_mtx = johnson(graph, directed=False, indices=indices)[:,indices]
    dist_mtx = johnson(subgraph, directed=False)

    # new center with shortest total distance to other points
    new_center = indices[np.argmin(dist_mtx.sum(axis=1))]

    return new_center

def update_centers_for_cluster_euclidean(data, cluster_assignments, cluster_idx):
    """Helper function of updating center of a given cluster
    Graph: connectivity graph. higher edge weight = higher distance
    """

    # all indices for points
    full_indices = np.array(range(data.shape[0]))

    # indices of points assigned to cluster index
    indices = full_indices[cluster_assignments[:,cluster_idx].astype(bool)]

    # data points in cluster
    cluster_points = data[indices,:]

    # distance matrix based on Euclidean or cosine distance
    dist_mtx = cdist(cluster_points, cluster_points, metric="euclidean")

    # new center with shortest total distance to other points
    new_center = indices[np.argmin(dist_mtx.sum(axis=1))]

    return new_center

class markov:

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
        self.embedding = None # embedding

        # model params
        self.verbose=verbose

    def initialize_kernel_jaccard_parallel(self, k):
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
            similarity_matrix_rows = parallel(delayed(jaccard_for_row)(sym_graph, row_sums, i) for i in tqdm.tqdm(range(self.n)))

        if self.verbose:
            print("Building similarity LIL matrix...")

        similarity_matrix = lil_matrix((self.n, self.n))
        for i in tqdm.tqdm(range(self.n)):
            similarity_matrix[i] = similarity_matrix_rows[i]

        if self.verbose:
            print("Constructing CSR matrix...")

        self.M = similarity_matrix.tocsr()
        self.G = (self.M > 0).astype(float)

    def initialize_kernel_rbf_parallel(self, k:int):
        """Initialize adaptive bandwith RBF kernel (as described in C-isomap)"""

        if self.verbose:
            print("Computing kNN graph...")

        # compute kNN and the distance from each point to its nearest neighbors
        knn_graph = kneighbors_graph(self.Y, k, mode="connectivity", include_self=True)
        knn_graph_distances = kneighbors_graph(self.Y, k, mode="distance", include_self=True)

        if self.verbose:
            print("Computing radius for adaptive bandwidth kernel...")

        # compute median distance for each point amongst k-nearest neighbors
        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            median_distances = parallel(delayed(kth_neighbor_distance)(knn_graph_distances, k//2, i) for i in tqdm.tqdm(range(self.n)))

        # convert to numpy array
        median_distances = np.array(median_distances)

        # take AND

        if self.verbose:
            print("Making graph symmetric...")
        sym_graph = (knn_graph + knn_graph.T > 0).astype(float)

        if self.verbose:
            print("Computing RBF kernel...")

        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            similarity_matrix_rows = parallel(delayed(rbf_for_row)(sym_graph, self.Y, median_distances, i) for i in tqdm.tqdm(range(self.n)))

        if self.verbose:
            print("Building similarity LIL matrix...")

        similarity_matrix = lil_matrix((self.n, self.n))
        for i in tqdm.tqdm(range(self.n)):
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

        order = np.argsort(-(np.real(w)))

        w = w[order]
        v = v[:,order]

        # set embedding
        lamb = np.power(np.real(w), t)
        self.embedding = np.real(v) @ np.diag(lamb)


    def set_distances(self, d):
        self.G = d

    def set_similarity(self, s):
        self.M = s

    def initialize_centers_uniform(self, k):
        # k = number of centers
        # for now just random uniform sample
        self.centers = np.random.choice(range(self.n), k, replace=False)

    def initialize_centers_cur(self, k:int, epsilon:float):
        """Implementation of row/column selection using leverage scores
        Inputs:
            k (int): desired rank
            epsilon (float): error tolerance
        """
        if self.verbose:
            print("Computing first %d singular vectors..." % k)

        # Note: per documentation, the order of singular values is not guaranteed
        # u, s, vt = svds(self.M, k=k)
        w, v = eigs(self.T, k=k, which="LM")
        vt = v.T

        if self.verbose:
            print("Computing leverage scores...")

        # compute square norms of right singular vectors
        lev_scores = np.square(np.linalg.norm(vt, axis=0))
        norm_lev_scores = (1./k) * lev_scores

        if self.verbose:
            print("Computing column probabilities...")

        # multiplicative constant that increases # columns selected
        c = k * np.log(k) / epsilon**2

        # compute column selection probabilities
        p = np.minimum(1., c * norm_lev_scores)

        if self.verbose:
            print("Selecting columns...")

        # sample
        selected = (np.random.binomial(1, p=p) > 0)

        # set selected colummns
        self.centers = self.indices[selected]

        if self.verbose:
            print("Selected %d columns!" % len(self.centers))

    def assign_clusters(self):
        """Assign clusters based on Markov absorption probabilities"""
        # transition matrix for nonabsorbing states
        nonabsorbing_states = np.array([idx not in self.centers for idx in self.indices])
        Q = self.T[:,nonabsorbing_states][nonabsorbing_states,:]

        # compute fundamental matrix 
        F = np.linalg.inv(np.eye(sum(nonabsorbing_states)) - Q)

        # compute absorption probabilities
        R = self.T[nonabsorbing_states,:][:,self.centers]
        B = F @ R

        assignments_nonabsorbing = (B == B.max(axis=1)).astype(int)

        self.assignments_bool = np.zeros((self.n, len(self.centers)))
        self.assignments_bool[nonabsorbing_states,:] = assignments_nonabsorbing
        self.assignments_bool[self.centers,:] = np.eye(len(self.centers))

        self.abs_probs = np.zeros((self.n, len(self.centers)))
        self.abs_probs[nonabsorbing_states,:] = B
        self.abs_probs[self.centers,:] = np.eye(len(self.centers))

        #self.assignments_bool = np.concatenate([np.eye(len(self.centers)), assignments_nonabsorbing], axis=0)

    def _compute_modularity_matrix(self):
        """Computes modularity matrix B to give an idea of quality of clustering
        
        B_{ij} = A_{ij} - (k_i k_j)/2m
        where A is the adjacency matrix
        and k_i is the number of edges connected to node i
        """
        # get (unweighted adjacency matrix)
        A = (self.G > 0).astype(float)

        # get k
        k = A.sum(axis=1)

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
        m = np.sum(A)

        # get boolean assignments matrix
        S = self.assignments_bool

        return 1./(2.*m) * np.trace(S.T @ A @ S - S.T @ k @ k.T @ S / (2.*m))

    def get_metacell_coordinates(self, coordinates=None):
        if coordinates is None:
            coordinates = self.Y

        return coordinates[self.centers,:]

    def get_smoothed_metacell_coordinates(self, coordinates=None):
        if coordinates is None:
            coordinates = self.Y

        #weights = np.multiply(self.assignments_bool, self.metacell_distances)
        # weights = self.assignments_bool
        weights = np.array(self.abs_probs)
        sum_weights = weights.sum(axis=0, keepdims=True)

        return np.array(weights.T @ coordinates / sum_weights.T)

    def get_cluster_mean_and_cov(self, coordinates, cluster_idx):
        """Given coordinates computes the gene expression mat
        Returns tuple of mean, cov
        """

        # identify cells corresponding to that cluster
        assignment_vector = self.assignments_bool[:,cluster_idx].astype(bool)

        # select coordinates associated with that cluster
        cluster_coordinates = coordinates[assignment_vector,:].toarray()

        # unit normalize
        normalized_coordinates = cluster_coordinates / cluster_coordinates.sum(axis=1, keepdims=True)
        #print(normalized_coordinates.sum(axis=1))

        # get mean (multinomial i guess)
        cluster_sum = np.sum(cluster_coordinates, axis=0)
        cluster_mean = cluster_sum / np.sum(cluster_sum)

        # get covariance
        cluster_cov = (normalized_coordinates - cluster_mean).T @ (normalized_coordinates - cluster_mean)
        return cluster_mean, cluster_cov

    def get_expected_mean_and_cov(self, coordinates, cluster_idx):
        """Basically the same as above, but computes expected covariance given the cluster mean"""
        mean, cov = self.get_cluster_mean_and_cov(coordinates, cluster_idx)
        mean_unsqueezed = mean.reshape(-1,1)
        expected_cov = - mean_unsqueezed @ mean_unsqueezed.T
        expected_cov = expected_cov - np.diag(np.diag(expected_cov)) + np.diag(mean * (1-mean))
        return mean, expected_cov

    def get_metacell_sizes(self):
        return np.sum(self.abs_probs, axis=0)

    def prune_small_metacells(self, thres:int=1):
        metacell_sizes = self.get_metacell_sizes()
        self.centers = self.centers[metacell_sizes >= thres]

    def prune_small_metacells_original(self, k:int=15):
        """Density-based pruning of metacells falling within adaptive bandwidth

        If a metacell is within certain distance of another metacell, remove it
        """
        # get radius for each metacell
        knn_distances = kneighbors_graph(self.embedding, k, mode="distance", include_self=False)

        # get max acceptable distance for each metacell
        radii = np.max(knn_distances, axis=1).toarray()[self.centers]

        # compute metacell distance matrix
        metacell_distances = cdist(self.embedding[self.centers,:], self.embedding[self.centers,:])

        print("Current number of metacells: %d" % len(self.centers))

        include = np.ones(len(self.centers)).astype(bool)
        for i in range(1,len(self.centers)):
            if np.min(metacell_distances[i,:i]) < radii[i]:
                include[i] = False

        print("New number of metacells: %d" % sum(include))
        
        # update centers
        self.centers = self.centers[include]

    def get_metacell_graph(self):
        """New connectivity graph for meta-cells"""
        return (self.assignments_bool.T @ self.G @ self.assignments_bool > 0).astype(float)

    def markov_clusters(self, min_size:int=20, max_iter:int=50, init_centers="cur", k=50, epsilon=1., max_centers=200):
        
        if init_centers is None:
            self.initialize_centers_uniform(max_centers)
        elif init_centers == "cur":
            self.initialize_centers_cur(k=k, epsilon=epsilon)

        self.assign_clusters()
        self.prune_small_metacells(thres=min_size)
        self.assign_clusters()
        self.prune_small_metacells(thres=min_size)
        self.assign_clusters()

        # it=1
        # converged=False

        # # coordinate system to use for k medoids
        # if self.embedding is None:
        #     data = self.Y
        # else:
        #     data = self.embedding

        # with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:

        #     while not converged and it <= max_iter:

        #         if self.verbose:
        #             print("Iteration %d of %d" % (it, max_iter))

        #         k = len(self.centers)

        #         # iterable
        #         cluster_it = tqdm.tqdm(range(k))

        #         new_centers = parallel(delayed(update_centers_for_cluster_euclidean)(data, 
        #             self.assignments_bool, cluster_idx) for cluster_idx in cluster_it)

        #         new_centers = np.array(new_centers)

        #         if np.all(new_centers == self.centers):
        #             converged=True
        #             print("Converged after %d iterations(s)!" % it)

        #         self.centers = new_centers

        #         # assign clusters
        #         self.assign_clusters_euclidean()
        #         #self.prune_small_metacells(thres=min_size)
        #         self.prune_small_metacells(k=min_size)
        #         self.assign_clusters_euclidean()

        #         it += 1

