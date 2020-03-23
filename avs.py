# Implementation of adaptive volume sampling

import numpy as np
from sklearn.neighbors import kneighbors_graph

# helper fx for computing projection matrix
def projection_matrix(A):
	"""Returns projection matrix P for subspace A"""
	return A @ np.linalg.inverse(A.T @ A) @ A.T

class avs:

	def __init__(self, Y, verbose:bool=False):

		# dimensionality of input
		self.n, self.d = M.shape

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

		self.M = jaccard_graph.toarray()

	def sample(self, k:int):

		S = np.zeros(self.n).astype(bool)
		E = self.M.copy()
		
		for it in range(k):

			if self.verbose:
				print("Beginning iteration %d" % it)
				print("Computing row probabilities...")

			# compute probability of selecting each row
			row_probabilities_raw = np.linalg.norm(E, axis=1) / np.linalg.norm(E)
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
			P = projection_matrix(self.M[S,:])

			# update E
			E = self.M - P @ self.M

		self.metacells = np.where(S)[0]

		return self.metacells







