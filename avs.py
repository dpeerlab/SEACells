# Implementation of adaptive volume sampling

import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix, csr_matrix, dok_matrix
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import norm

# helper fx for computing projection matrix
def projection_matrix(A):
	"""Returns projection matrix P for subspace A (A is sparse)"""
	if np.ndim == 1:
		A = A.reshape(1,-1)
	return csr_matrix(A @ np.linalg.inv((A.T @ A).toarray()) @ A.T)

class avs:

	def __init__(self, Y, verbose:bool=False):

		# dimensionality of input
		self.n, self.d = Y.shape

		# input data
		self.Y = Y
		self.verbose = verbose

	def initialize_kernel(self, k):
		"""Set Jaccard kernel"""

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

		self.M = csr_matrix(jaccard_graph)

	def sample(self, k:int):

		S = np.zeros(self.n).astype(bool)
		E = self.M.copy()
		
		for it in range(k):

			if self.verbose:
				print("Beginning iteration %d" % it)
				print("Computing row probabilities...")

			# compute probability of selecting each row
			row_probabilities_raw = norm(E, axis=1) / norm(E)
			row_probabilities_norm = row_probabilities_raw / sum(row_probabilities_raw)

			if self.verbose:
				print("Sampling new index...")

			# sample new row index
			row_idx = np.random.choice(range(self.n), 1, p=row_probabilities_norm)

			if self.verbose:
				print("Selected index %d" % row_idx[0])

			# add selected index
			S[row_idx] = True

			if self.verbose:
				print("Computing projection matrix...")

			# compute projection matrix for subspace
			P = projection_matrix(self.M[S,:].T)

			# update E
			E = self.M - P @ self.M

		self.metacells = np.where(S)[0]

		if self.verbose:
			print("Computing metacell assignments...")

		self.compute_metacell_assignments()

		return self.metacells

	def compute_metacell_assignments(self):
		"""Return metacell assignment (int) for each data point in Y"""
		if self.metacells is None:
			print("Metacells not computed yet.")
		else:
			dpp_distances = cdist(self.Y, self.Y[self.metacells,:])
			dpp_clusters = np.argmax(dpp_distances, axis=1)
			dpp_boolean = csr_matrix((dpp_distances == dpp_distances.min(axis=1)[:,None]).astype(int))

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
		weights = self.metacell_boolean.multiply(self.M[self.metacells,:].T)
		return np.array((weights.T @ coordinates) / weights.sum(axis=0).T)

	def compute_metacell_connectivities(self):
		"""Return (binary) adjacency matrix of metacells"""
		# TODO: return sparse matrix?
		return (self.metacell_boolean.T @ self.M @ self.metacell_boolean > 0).astype(float)

	def get_metacell_sizes(self):
		return np.array(self.metacell_boolean.sum(axis=0)).ravel()







