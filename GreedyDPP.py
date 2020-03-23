import numpy as np
from scipy.sparse import coo_matrix, dok_matrix
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph

# TODO: better selection of initial point?

class GreedyDPP:

	def __init__(self, Y, verbose=False):
		"""
		Inputs:
			L: kernel matrix
		"""
		# coordinates
		self.Y = Y

		# number of cells
		self.n = Y.shape[0]
		self.verbose = verbose
		self.metacells = None

	def initialize_kernel(self, r):
		"""Compute kernel with adaptive bandwidth given coordinates Y
		Inputs:
			- r: rth nearest neighbor determines the bandwidth of the kernel
		TODO: adapt for sparse matrix
		"""
		dist_mtx = cdist(self.Y, self.Y)**2.

		# find rth closest point for each element
		bw_per_cell = np.expand_dims(dist_mtx[np.argsort(np.argsort(dist_mtx, axis=1), axis=1) == r], -1)**(-0.5)
		bw = bw_per_cell @ bw_per_cell.T

		self.L = np.exp(np.multiply(-dist_mtx, bw))

	def initialize_sparse_kernel(self, k, r):
		"""Compute sparse RBF kernel
		Inputs:
			k: number of nearest neighbors in kNN graph
			r: nearest neighbor used to determine width of adaptive kernel radius
		"""

		# initialize RBF kernel with adaptive bandwidth
		if self.verbose:
			print("Initializing RBF kernel...")
		self.initialize_kernel(r)

		# get knn graph
		if self.verbose:
			print("Computing kNN graph...")
		knn_graph = kneighbors_graph(self.Y, k, include_self=True).toarray()

		# take AND of kNN graph to make it symmetric
		knn_sym = np.multiply(knn_graph, knn_graph.T)

		# weight edges with RBF kernel
		self.L = np.multiply(self.L, knn_sym)

	def initialize_jaccard_kernel(self, k):
		"""Set Jaccard kernel"""

		if self.verbose:
			print("Computing kNN graph...")

		knn_graph = kneighbors_graph(self.Y, k, include_self=True)

		# all neighbors within path length two
		path_len_2 = knn_graph @ knn_graph

		if self.verbose:
			print("Making graph symmetric...")

		# symmetrize graph using OR
		sym_graph = (path_len_2 + path_len_2.T > 0).astype(int)

		# compute row sums
		row_sums = sym_graph.sum(axis=1)

		# compute weights using Jaccard similarity
		jaccard_graph = coo_matrix(sym_graph)

		if self.verbose:
			print("Computing intersections...")

		# compute intersections
		intersections = dok_matrix(sym_graph @ sym_graph.T)

		if self.verbose:
			print("Computing Jaccard similarity for %d pairs..." % len(jaccard_graph.data))
		for v, (i,j) in enumerate(zip(jaccard_graph.row, jaccard_graph.col)):
			intersection = intersections[i,j]
			jaccard_graph.data[v] = intersection / (row_sums[i,0] + row_sums[j,0] - intersection)

		self.L = jaccard_graph.toarray()

	def initialize_sparse_jaccard(self, k):
		"""PhenoGraph-like Jaccard kernels
		Entries are non-zero iff i,j are neighbors"""
		if self.verbose:
			print("Computing kNN graph...")
		knn_graph = kneighbors_graph(self.Y, k, include_self=True)

		if self.verbose:
			print("Making matrix symmetric...")
		# symmetrize kNN graph using OR operation
		sym_knn_graph = (knn_graph + knn_graph.T > 0).astype(float)

		# compute row sums
		row_sums = sym_knn_graph.sum(axis=1)

		# compute weights using Jaccard similarity
		jaccard_graph = coo_matrix(sym_knn_graph)

		# intersections
		if self.verbose:
			print("Computing set intersections...")
		intersections = dok_matrix(coo_matrix(sym_knn_graph @ sym_knn_graph.T))

		if self.verbose:
			print("Computing Jaccard similarity for %d pairs..." % len(jaccard_graph.data))
		for v, (i, j) in enumerate(zip(jaccard_graph.row, jaccard_graph.col)):
			intersection = intersections[i,j]
			jaccard_graph.data[v] = intersection / (row_sums[i,0] + row_sums[j,0] - intersection)

		self.L = jaccard_graph.toarray()

	def compute_log_determinant(self, d_j, curr_log_det):
		"""Fast computation of matrix determinant given previous determinant
		Inputs:
			- d_j (diagonal entry of cholesky decomp being added)
			- curr_log_det (current log determinant)
		"""
		return np.log(d_j) + curr_log_det

	def sample_unconstrained(self):
		"""Greedily adds points until determinant starts to decrease"""
		if self.verbose:
			print("Finding metacells...")

		# stores last row of cholesky decomposition of subset kernel matrix
		c = np.zeros((self.n, self.n))

		# stores contribution of each point to the determinant
		d = np.diag(self.L).copy()

		# stores indices of currently selected metacells
		Yg = set()

		# whether or not to keep adding metacells
		cont = True
		it = 0

		while cont and it < self.n:
			if self.verbose:
				print("Beginning iteration %d" % it)

			# find current max
			j = np.argmax(d)

			# check to see if we should keep going
			# update selected indices if appropriate
			if d[j] < 1.0:
				# keep going only if determinant is non-decreasing
				cont = False
			else:
				Yg.add(j)

			# update e, c, and d
			e_n = (self.L[:,j] - c @ c[j,:].T)/d[j]
			c[:,it] = e_n
			d = d - np.square(e_n)

			# remove d[j] so it won't be chosen again in future iterations
			d[j] = 0.

			# update iteration count
			it += 1

		self.metacells = np.array(list(Yg))

		if self.verbose:
			print("Finished!")
			print("Found %d metacells" % len(self.metacells))
			print("Computing metacell assignments...")

		self.compute_metacell_assignments()
		return self.metacells


	def sample_k(self, k:int, kernel="L"):
		"""Greedy MAP approximation of DPP parameterized by L of size k
		Inputs:
			L (numpy array): n x n, where n is the number of samples
			k: size of desired subset
		Returns: None
			(but updates the metacell centers and metacell assignments of data points)
		"""
		print("Finding metacells...")
		c = np.zeros((self.n,k))
		d = np.diag(self.L).copy()
		Yg = set()
		curr_log_det = 0.

		for i in range(k):
			if self.verbose:
				print("Iteration %d of %d" % (i, k))

			# find max
			j = np.argmax(d)

			if self.verbose:
				print("Selected %d" % j)
				print("Delta log determinant: %.4e" % d[j])

			# append j to output
			Yg.add(j)

			# update e, c, and d
			e_n = (self.L[:,j] - c @ c[j,:].T)/d[j]
			c[:,i] = e_n
			d = d - np.square(e_n)

			# remove dj so it won't be chosen again
			d[j] = 0.

		print("Finished!")
		self.metacells = np.array(list(Yg))
		self.compute_metacell_assignments()
		return self.metacells

	def prune_small_metacells(self, thres=1):
		"""Get rid of metacells with < 5 cells (or something)"""
		metacell_sizes = self.get_metacell_sizes()
		below_thres = metacell_sizes < thres

		self.metacells = self.metacells[~below_thres]
		self.compute_metacell_assignments()

	def compute_metacell_assignments(self):
		"""Return metacell assignment (int) for each data point in Y"""
		if self.metacells is None:
			print("Metacells not computed yet.")
		else:
			dpp_distances = self.L[:,self.metacells]
			dpp_clusters = np.argmax(dpp_distances, axis=1)
			dpp_boolean = (dpp_distances == dpp_distances.max(axis=1)[:,None]).astype(int)

			# set stuff
			self.metacell_assignments = dpp_clusters
			self.metacell_boolean = dpp_boolean

	def compute_metacell_assignments_euclidean(self):
		"""Return metacell assignment (int) for each data point in Y"""
		if self.metacells is None:
			print("Metacells not computed yet.")
		else:
			dpp_distances = cdist(self.Y, self.Y[self.metacells,:])
			dpp_clusters = np.argmin(dpp_distances, axis=1)
			dpp_boolean = (dpp_distances == dpp_distances.min(axis=1)[:,None]).astype(int)

			# set stuff
			self.metacell_assignments = dpp_clusters
			self.metacell_boolean = dpp_boolean

	def compute_metacell_coordinates(self, coordinates=None):
		"""Return metacell coordinates (arithmetic average of cells assigned to the metacell)"""
		if self.metacells is None:
			print("Metacells not computed yet.")
			return
		if coordinates is None:
			coordinates = self.Y
		weights = np.multiply(self.metacell_boolean, self.L[:,self.metacells])
		return (weights.T @ coordinates) / weights.sum(axis=0, keepdims=True).T

	def compute_metacell_connectivities(self):
		"""Return (binary) adjacency matrix of metacells"""
		# TODO: return sparse matrix?
		return (self.metacell_boolean.T @ self.L @ self.metacell_boolean > 0).astype(float)

	def compute_weighted_metacell_connectivities(self, r):
		"""Sparse connectivities weighted with adaptive kernel"""
		connectivities = self.compute_metacell_connectivities()

		# RBF kernel weights
		coords = self.compute_metacell_coordinates()
		dist_mtx = cdist(coords, coords)**2.

		# find rth closest point for each element
		bw_per_cell = np.expand_dims(dist_mtx[np.argsort(np.argsort(dist_mtx, axis=1), axis=1) == r], -1)**(-0.5)
		bw = bw_per_cell @ bw_per_cell.T

		adaptive_rbf = np.exp(np.multiply(-dist_mtx, bw))
		return np.multiply(adaptive_rbf, connectivities)

	def get_metacell_sizes(self):
		return self.metacell_boolean.sum(axis=0)