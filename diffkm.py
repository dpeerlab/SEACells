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

def weighted_jaccard_for_row(G, i):
    Gikron = kron(np.ones((G.shape[0], 1)), G[i,:])
    abs_of_diff = (G - Gikron).tocsr()
    abs_of_diff.data = np.abs(abs_of_diff.data)
    
    row_min = 0.5 * (G + Gikron - abs_of_diff)

    sum_row_min = coo_matrix(row_min.sum(axis=1).T)
    #print(sum_row_min.col)
    row_max = 0.5 * (G[sum_row_min.col] + Gikron[sum_row_min.col] + abs_of_diff[sum_row_min.col])
    row_max_sums = np.array(row_max.sum(axis=1)).ravel()
    #sum_row_max = coo_matrix(row_max.sum(axis=1).T)
    #print(sum_row_min)
    #print(sum_row_max)
    sum_row_min.data = sum_row_min.data / row_max_sums

    out = sum_row_min#sum_row_min.multiply(sum_row_max)
    return out

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
        #one_nn_graph_distances = kneighbors_graph(self.Y, 1, mode="distance", include_self=False)

        # mean distance (smallest thing possible so that the graph remains connected)
        #max_distance = -np.min(-one_nn_graph_distances.max(axis=1).toarray())

        # radius neighbors - distances
        #rn_graph = radius_neighbors_graph(self.Y, mode="connectivity", include_self=True, radius=max_distance, n_jobs=-1)
        #rn_graph = rn_model.fit_transform(self.Y)

        if self.verbose:
            print("Computing radius for adaptive bandwidth kernel...")

        # compute median distance for each point amongst k-nearest neighbors
        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            # median = k // 2
            median = k // 3
            median_distances = parallel(delayed(kth_neighbor_distance)(knn_graph_distances, median, i) for i in tqdm(range(self.n)))

        # convert to numpy array
        median_distances = np.array(median_distances)

        # take AND

        if self.verbose:
            print("Making graph symmetric...")
        sym_graph = (knn_graph + knn_graph.T > 0).astype(float)
        # sym_graph = knn_graph

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
        self.G = (self.M > 0).astype(float)
        return self.M

    def initialize_kernel_jaccard_parallel(self, k:int):
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

    def initialize_kernel_weighted_jaccard_parallel(self, k:int):
        """Uses Jaccard similarity between nearest neighbor sets as PSD kernel"""
        
        sym_graph = self.initialize_kernel_rbf_parallel(k)
        row_sums = np.sum(sym_graph, axis=1)

        if self.verbose:
            print("Computing Jaccard similarity...")

        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            similarity_matrix_rows = parallel(delayed(weighted_jaccard_for_row)(sym_graph, i) for i in tqdm(range(self.n)))

        self.M = vstack(similarity_matrix_rows).tocsr()
        # if self.verbose:
        #     print("Building similarity LIL matrix...")

        # similarity_matrix = lil_matrix((self.n, self.n))
        # for i in tqdm(range(self.n)):
        #     similarity_matrix[i] = similarity_matrix_rows[i]

        # if self.verbose:
        #     print("Constructing CSR matrix...")

        # self.M = similarity_matrix.tocsr()
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
        self.eigenvectors = v

    ##############################################################
    # Clustering and sampling
    ##############################################################

    def kmpp(self, k):
        """kmeans++ initialization for medoids

        k: number of centroids
        """
        # array for storing centroids
        self.centers = np.zeros(k, dtype=int)

        # select initial point randomly
        new_point = np.random.choice(range(self.n), 1)
        self.centers[0] = new_point

        # initialize min distances
        distances = cdist(self.embedding, self.embedding[new_point, :].reshape(1,-1), metric="euclidean")

        # assign rest of points
        for ix in range(1, k):
            new_point = np.argmax(distances.ravel())
            self.centers[ix] = new_point

            # get distance from all poitns to new points
            new_point_distances = cdist(self.embedding, self.embedding[new_point, :].reshape(1,-1), metric="euclidean")

            # update min distances
            combined_distances = np.hstack([distances, new_point_distances])

            distances = np.min(combined_distances, axis=1, keepdims=True)

    def adaptive_volume_sampling(self, t:int=0, max_cols:int=1000):
        """Fast greedy adaptive CSSP

        From https://arxiv.org/pdf/1312.6838.pdf
        """
        A = self.T.copy()
        #A = A.dot(A.dot(A))

        print("Initializing residual matrix...")

        # exponentiate
        for _ in range(t):
            print("Exponent: %d" % _)
            A = A.dot(self.T)

        # save A
        self.A = A

        # precomute ATA (this is actually bad idea, results in very dense matrix)
        # solution is just to compute f and g directly
        # ATA = A.T.dot(A)
        # can probably parallelize this
        print("Initializing f and g...")
        #At = A.T.tocsr()
        Acsc = A.tocsc()
        f = np.zeros(self.n)
        #g = np.zeros(self.n)
        for ix in tqdm(range(self.n)):
            #print(ix)
            f[ix] = np.sum((A.T.dot(Acsc[:,ix])).data**2)
            #print(ix)
            #g[ix] = norm(Acsc[:,ix])**2

        #print(f)
        g = np.array(norm(Acsc, axis=0)**2).ravel()
        #print(g)


        k = max_cols

        d = np.zeros((k, self.n))
        omega = np.zeros((k, self.n))

        # initially best score?
        initial_best = np.max(f/g)
        #print("Initial best score: ", initial_best)

        # keep track of selected indices
        S = np.zeros(k, dtype=int)

        # sampling
        for j in tqdm(range(k)):

            # select point
            score = f/g
            p = np.argmax(score)

            # print residuals
            residual = np.sum(f)
            #print("Current best: ", score[p])

            # delta_term1 = (ATA[:,p]).toarray().ravel()
            delta_term1 = (A.T @ Acsc[:,p]).toarray().ravel()
            delta_term2 = np.multiply(omega[:,p].reshape(-1,1), omega).sum(axis=0)
            delta = delta_term1 - delta_term2

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

            # ATAo = (ATA @ o.reshape(-1,1)).ravel()
            ATAo = A.T.dot(A.dot(o.reshape(-1,1))).ravel()
            term2 = np.multiply(o, ATAo - pl)

            # update f
            f = (f - 2 * term2 + term1)

            # update g
            g = g + omega_hadamard

            # store omega and delta
            d[j,:] = delta
            omega[j,:] = o

            # add index
            S[j] = p

        # store everything...
        self.S = S

    def binary_search(self, epsilon):

        max_cols = len(self.S)
        S = self.S.copy()

        # find centers by checking approximation error (use binary search)
        min_ix = 0
        max_ix = max_cols

        # get original matrix norm
        original_norm = norm(self.A)
        Acsc = self.A.tocsc()
        
        while max_ix - min_ix > 1:

            # current is midpoint of max and min
            curr = (max_ix - min_ix) // 2 + min_ix
            centers = S[:curr]
            proj = get_projection(self.A, Acsc[:, centers].tocsr())

            # get projection error
            err = norm(self.A - proj)

            print("Current midpoint: ", curr)
            print("Error: ", err / original_norm)

            if err/original_norm >= epsilon:
                min_ix = curr
            else:
                max_ix = curr

        print("Search complete. Required metacells: ", curr)

        self.centers = S[:curr]

    def adaptive_volume_sampling_original(self, k:int, t:int=0):
        """Fast greedy adaptive CSSP

        From https://arxiv.org/pdf/1312.6838.pdf
        """
        A = self.T.copy()
        #A = A.dot(A.dot(A))

        print("Initializing residual matrix...")

        # exponentiate
        for _ in range(t):
            print("Exponent: %d" % _)
            A = A.dot(self.T)

        # save A
        self.A = A

        # precomute ATA (this is actually bad idea, results in very dense matrix)
        # solution is just to compute f and g directly
        # ATA = A.T.dot(A)
        # can probably parallelize this
        print("Initializing f and g...")
        Acsc = A.tocsc()
        f = np.zeros(self.n)
        g = np.zeros(self.n)
        for ix in tqdm(range(self.n)):
            #print(ix)
            f[ix] = norm(A.T.dot(Acsc[:,ix].tocsr()))**2
            #print(ix)
            g[ix] = norm(Acsc[:,ix])**2

        print(f)
        print(g)

        # initialization
        # f = np.power(norm(ATA, axis=0), 2)
        # g = ATA.diagonal()

        d = np.zeros((k, self.n))
        omega = np.zeros((k, self.n))

        # initially best score?
        initial_best = np.max(f/g)
        #print("Initial best score: ", initial_best)

        # keep track of selected indices
        S = set([])

        # sampling
        for j in tqdm(range(k)):

            # select point
            score = f/g
            p = np.argmax(score)

            # print residuals
            residual = np.sum(f)
            #print("Current best: ", score[p])

            # delta_term1 = (ATA[:,p]).toarray().ravel()
            delta_term1 = (A.T @ Acsc[:,p]).toarray().ravel()
            delta_term2 = np.multiply(omega[:,p].reshape(-1,1), omega).sum(axis=0)
            delta = delta_term1 - delta_term2

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

            # ATAo = (ATA @ o.reshape(-1,1)).ravel()
            ATAo = A.T.dot(A.dot(o.reshape(-1,1))).ravel()
            term2 = np.multiply(o, ATAo - pl)

            # update f
            f = (f - 2 * term2 + term1)

            # update g
            g = g + omega_hadamard

            # store omega and delta
            d[j,:] = delta
            omega[j,:] = o

            # add index
            S.add(p)

        self.centers = np.array(list(S))

    def sum_sq_distances(self):
        distances = cdist(self.embedding, self.embedding[self.centers,:], metric="euclidean")
        return np.sum(np.min(distances**2, axis=1))

    def assign_hard_clusters(self):
        """Use k-medoids to assign hard cluster labels"""
        distances = cdist(self.embedding, self.embedding[self.centers,:], metric="euclidean")
        self.assignments = np.argmin(distances, axis=1)

    def get_new_centers(self):
        """Wrapper for updating all cluster centers in parallel"""
        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            new_centers = parallel(delayed(get_new_center_for_cluster)(self.embedding, self.assignments, i) for i in tqdm(range(len(self.centers))))
        
        return np.array(new_centers, dtype=int)

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

    def cluster(self, k:int, t:int, max_iter:int=200, epsilon:float=0.01):
        """Wrapper for running adaptive volume sampling, then assigning each cluster.
        """
        # initialize with km++
        # self.kmpp(k)

        # initialize with random points...
        # self.centers = np.random.choice(range(self.n), k, replace=False)

        # initialize with AVS
        if hasattr(self, "S"):
            self.binary_search(epsilon)
        else:
            self.adaptive_volume_sampling(t, k)
            self.binary_search(epsilon)
        
        # self.centers = np.random.choice(range)

        # assign clusters
        # self.assign_hard_clusters()

        # converged=False
        # it = 0
        # while not converged and it < max_iter:
        #     # update iteration count
        #     it += 1

        #     # update centers
        #     new_centers = self.get_new_centers()
        #     #print(self.centers)
        #     #print(new_centers)

        #     # update assignments
        #     self.assign_hard_clusters()

        #     # check convergence (no assignments updated --> objectve local max)
        #     #print(self.centers)
        #     #print(new_centers)
        #     if np.all(np.equal(self.centers, new_centers)):
        #         converged=True
        #         print("Converged after %d iterations!" % it)
        #     else:
        #         self.centers = new_centers.copy()

        # print("Sum of squared distances: ", self.sum_sq_distances())

        # self.assign_soft_clusters()

        # self.W = self.A[:,self.centers]
        self.W = np.array(self.A[:,self.centers] / self.A[:,self.centers].sum(axis=1))
        # self.W = np.array(self.A[self.centers,:])
        #self.W = np.abs(projection_w(self.A, self.Acsc[:, self.centers]))
        #self.W = self.W / self.W.sum(axis=1, keepdims=True)

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
        """Exponent parameter is for softmax tempering..."""
        if coordinates is None:
            coordinates = self.Y
        W = np.power(self.W, exponent)
        #W = W / W.sum(axis=0, keepdims=True)
        W = self.A[self.centers,:].toarray()
        return W.T @ coordinates

    ##############################################################
    # Label transfer
    ##############################################################

    def get_metacell_labels(self, labels, exponent=1.):
        """Given labels of original cells, transfer labels
        Exponent: optional softmax tempering
        """
        # get onehot encoding of labels
        unique_labels = set(labels)
        label2idx = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label = {idx:label for label,idx in label2idx.items()}

        # print(label2idx)

        # onehot labels
        onehot_labels = np.zeros((self.n, len(unique_labels)))
        label_indices = np.array([label2idx[label] for label in labels])
        onehot_labels[np.arange(self.n), label_indices] = 1.

        # print(onehot_labels)

        # get soft label assignments
        W = np.power(self.W, exponent)
        W = W / W.sum(axis=0, keepdims=True)
        metacell_labels = self.W.T @ onehot_labels

        # print(metacell_labels)

        # get hard labels
        metacell_hard_labels = np.argmax(metacell_labels, axis=1)

        # print(metacell_hard_labels)

        # get actual word labels
        return [idx2label[idx] for idx in metacell_hard_labels]



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