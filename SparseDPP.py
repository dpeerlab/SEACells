import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, dok_matrix
from scipy.sparse.linalg import norm
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph

class dpp:

	def __init__(self, Y, verbose):
		"""Initialize model parameters"""

		# number of samples and observation dim
		self.n, self.d = Y.shape

		# store data points
		self.Y = Y

		# verbosity
		self.verbose = True

	def initialize_kernel(self, k):
		"""Use Jaccard kernel from kNN graph.
		Nodes weighted by L2-norm 
		User must specify k"""
		if self.verbose:
			print("Computing kNN graph...")

		knn_graph = kneighbors_graph(self.Y, k, include_self=True)

		# all neighbors within path length two
		path_len_2 = knn_graph @ knn_graph

		if self.verbose:
			print("Making graph symmetric...")

		# symmetrize graph using OR
		sym_graph = (path_len_2 + path_len_2.T > 0).astype(float)

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
			jaccard_graph.data[v] = intersection / float(row_sums[i,0] + row_sums[j,0] - intersection)

		# convert jaccard graph to csr matrix
		jaccard_graph = csr_matrix(jaccard_graph)

		# compute row norms
		row_norms = csr_matrix(np.diag(norm(jaccard_graph, axis=1)))

		self.L = jaccard_graph

		self.M = jaccard_graph

		#self.M = row_norms * jaccard_graph * row_norms

	def set_kernel(self, M):
		"""Set precomputed kernel"""
		self.M = csr_matrix(M)

	def sample(self, max_iter = None):
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
		print(d.shape)

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

			# check to see if we should keep going
			# update selected indices if appropriate
			if d[0,j] < 1.0:
				# keep going only if determinant is non-decreasing
				cont = False
			else:
				Yg[j] = True

			# update e, c, and d
			e_n = (self.M[j,:].toarray() - c @ c[j,:].T)/d[0,j]
			c[:,it] = e_n
			d = d - np.square(e_n)

			# remove d[j] so it won't be chosen again in future iterations
			d[0,j] = 0.

			# update iteration count
			it += 1

		self.metacells = Yg
		self.compute_metacell_assignments_euclidean()
		return self.metacells

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
		weights = np.multiply(self.metacell_boolean, self.M[self.metacells,:].todense().T)
		return np.array((weights.T @ coordinates) / weights.sum(axis=0).T)

	def compute_metacell_connectivities(self):
		"""Return (binary) adjacency matrix of metacells"""
		# TODO: return sparse matrix?
		return (self.metacell_boolean.T @ self.M @ self.metacell_boolean > 0).astype(float)

	def get_metacell_sizes(self):
		return np.array(self.metacell_boolean.sum(axis=0)).ravel()