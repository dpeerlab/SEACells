# An implementation of of the Partioning Around Medoids (PAM) algorithm on a graph
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, dok_matrix, lil_matrix, diags
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import johnson, dijkstra
from scipy.sparse.linalg import svds, eigs, eigsh
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

# for parallelizing stuff
from multiprocessing import cpu_count, Pool
from contextlib import closing
from joblib import Parallel, delayed
from itertools import repeat, starmap
import tqdm

# get number of cores for multiprocessing
NUM_CORES = cpu_count()

#################################################
# Helper functions parallelization
#
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

class pam:

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

    def initialize_centers_dpp(self, max_iter:int=200):
        """Greedily adds points until determinant starts to decrease"""
        if self.verbose:
            print("Finding metacells...")

        # store max iter
        if max_iter is None:
            max_iter = self.n

        # stores last row of cholesky decomposition of subset kernel matrix
        c = np.zeros((self.n, self.n))

        # stores contribution of each point to the determinant
        d = self.M.diagonal().reshape(1,-1)

        # stores indices of currently selected metacells
        Yg = np.zeros(self.n).astype(bool)

        # whether or not to keep adding metacells
        cont = True
        it = 0

        while cont and it < max_iter:
            if self.verbose:
                print("Beginning iteration %d" % it)

            # find current max
            j = np.argmax(d)

            # update selected indices if appropriate
            Yg[j] = True

            # update e, c, and d
            e_n = (self.M[j,:].toarray() - c @ c[j,:].T)/d[0,j]
            c[:,it] = e_n
            d = d - np.square(e_n)

            # remove d[j] so it won't be chosen again in future iterations
            d[0,j] = 0.

            # update iteration count
            it += 1

        self.centers = self.indices[Yg]

    def initialize_centers_cur(self, k:int, epsilon:float):
        """Implementation of row/column selection using leverage scores
        Inputs:
            k (int): desired rank
            epsilon (float): error tolerance
        """
        if self.verbose:
            print("Computing first %d singular vectors..." % k)

        # Note: per documentation, the order of singular values is not guaranteed
        u, s, vt = svds(self.M, k=k)

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
        # find distance from each point to each center
        dist_mtx = johnson(self.G, directed=False, indices = self.centers).T

        # assign points to closest
        self.assignments = np.argmin(dist_mtx, axis=1)

        # save distances
        # self.metacell_distances = np.exp(-dist_mtx)
        self.metacell_distances = 1. / (dist_mtx + 1.)

        # boolean
        self.assignments_bool = (dist_mtx == dist_mtx.min(axis=1)[:,None]).astype(int)

    def assign_clusters_euclidean(self, coordinates=None):
        if coordinates is None:
            if self.embedding is None:
                coordinates = self.Y
            else:
                coordinates = self.embedding

        dist_mtx = cdist(coordinates, coordinates[self.centers,:], metric="euclidean")

        # assign points to closest
        self.assignments = np.argmin(dist_mtx, axis=1)

        # save distances
        self.metacell_distances = np.exp(-dist_mtx)

        # boolean
        self.assignments_bool = (dist_mtx == dist_mtx.min(axis=1)[:,None]).astype(int)


    def update_centers(self, k):

        converged = True

        for clust in range(k):
            # get current center
            curr_center = self.centers[clust]

            if self.verbose:
                print("Processing cluster %d" % clust)
                print("Current center: %d" % curr_center)
            assigned_points = self.indices[self.assignments_bool[:,clust].astype(bool)]

            # compute distances
            dist_mtx = dijkstra(self.G, directed=False, indices = assigned_points)[:,assigned_points]

            # pick the point with the minimum total distance to all points
            new_center = assigned_points[np.argmin(dist_mtx.sum(axis=1))]

            if self.verbose:
                print("New center: %d" % new_center)

            # check if update
            if curr_center != new_center:
                converged = False

            # update center
            self.centers[clust] = new_center

        return converged

    def get_metacell_coordinates(self, coordinates=None):
        if coordinates is None:
            coordinates = self.Y

        return coordinates[self.centers,:]

    def get_smoothed_metacell_coordinates(self, coordinates=None):
        if coordinates is None:
            coordinates = self.Y

        weights = np.multiply(self.assignments_bool, self.metacell_distances)
        sum_weights = weights.sum(axis=0, keepdims=True)

        return np.array(weights.T @ coordinates / sum_weights.T)

    def get_metacell_sizes(self):
        return np.sum(self.assignments_bool, axis=0)

    def prune_small_metacells(self, thres:int=1):
        metacell_sizes = self.get_metacell_sizes()
        self.centers = self.centers[metacell_sizes >= thres]

    def get_metacell_graph(self):
        """New connectivity graph for meta-cells"""
        return (self.assignments_bool.T @ self.G @ self.assignments_bool > 0).astype(float)

    def k_medoids(self, min_size:int=20, max_iter:int=10, init_centers="cur", k=50, epsilon=1., max_centers=200):

        # pick k centers
        #self.initialize_centers(k)
        if init_centers is None:
            self.initialize_centers_uniform(max_centers)
        elif init_centers == "dpp":
            self.initialize_centers_dpp(max_iter=max_centers)
        elif init_centers == "cur":
            self.initialize_centers_cur(k=k, epsilon=epsilon)


        k = len(self.centers)

        # count iterations
        it = 0

        converged = False

        while not converged and it < max_iter:

            if self.verbose:
                print("Iteration %d of %d" % (it, max_iter))

            # update iteration count
            it += 1

            # assign clusters
            self.assign_clusters()

            # prune small metacells
            self.prune_small_metacells(thres=min_size)

            if len(self.centers) != k:
                k = len(self.centers)
                self.assign_clusters()

            # update location of medoids
            converged = self.update_centers(k)
            if converged:
                print("Converged after %d iterations(s)!" % it)


    def k_medoids_parallel(self, min_size:int=20, max_iter:int=50, init_centers="cur", k=50, epsilon=1., max_centers=200):
        
        if init_centers is None:
            self.initialize_centers_uniform(max_centers)
        elif init_centers == "dpp":
            self.initialize_centers_dpp(max_iter=max_centers)
        elif init_centers == "cur":
            self.initialize_centers_cur(k=k, epsilon=epsilon)

        self.assign_clusters_euclidean()
        self.prune_small_metacells(thres=min_size)

        it=1
        converged=False

        # coordinate system to use for k medoids
        if self.embedding is None:
            data = self.Y
        else:
            data = self.embedding

        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:

            while not converged and it <= max_iter:

                if self.verbose:
                    print("Iteration %d of %d" % (it, max_iter))

                k = len(self.centers)

                # iterable
                cluster_it = tqdm.tqdm(range(k))

                new_centers = parallel(delayed(update_centers_for_cluster_euclidean)(data, 
                    self.assignments_bool, cluster_idx) for cluster_idx in cluster_it)

                new_centers = np.array(new_centers)

                if np.all(new_centers == self.centers):
                    converged=True
                    print("Converged after %d iterations(s)!" % it)

                self.centers = new_centers

                # assign clusters
                self.assign_clusters_euclidean()
                self.prune_small_metacells(thres=min_size)
                self.assign_clusters_euclidean()

                it += 1

