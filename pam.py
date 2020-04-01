# An implementation of of the Partioning Around Medoids (PAM) algorithm on a graph
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, dok_matrix
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import johnson, dijkstra
from scipy.sparse.linalg import svds

class pam:

	def __init__(self, Y, verbose:bool=False):
		"""Initialize model parameters"""
		# data parameters
		self.n, self.d = Y.shape

		# indices of each point
		self.indices = np.array(range(self.n))

		# save data
		self.Y = Y

		# model params
		self.verbose=verbose

	def initialize_kernel_jaccard(self, k):
		"""Use Jaccard kernel from kNN graph.
		Nodes weighted by L2-norm 
		User must specify k"""
		if self.verbose:
			print("Computing kNN graph...")

		knn_graph = kneighbors_graph(self.Y, k, include_self=False)

		# take AND
		sym_graph = (np.multiply(knn_graph, knn_graph.T) > 0).astype(float)

		# all neighbors within path length two
		path_len_2 = sym_graph @ sym_graph

		if self.verbose:
			print("Making graph symmetric...")

		# compute row sums
		row_sums = sym_graph.sum(axis=1)

		# compute weights using Jaccard similarity
		jaccard_graph = coo_matrix(path_len_2)
		similarity_matrix = coo_matrix(path_len_2)

		if self.verbose:
			print("Computing intersections...")

		# compute intersections
		intersections = dok_matrix(sym_graph @ sym_graph.T)

		if self.verbose:
			print("Computing Jaccard similarity for %d pairs..." % len(jaccard_graph.data))
		for v, (i,j) in enumerate(zip(jaccard_graph.row, jaccard_graph.col)):
			intersection = intersections[i,j]
			jaccard_sim = intersection / float(row_sums[i,0] + row_sums[j,0] - intersection)
			similarity_matrix.data[v] = jaccard_sim
			jaccard_graph.data[v] = 1 - jaccard_sim

		# convert jaccard graph to csr matrix
		jaccard_graph = csr_matrix(jaccard_graph)
		similarity_matrix = csr_matrix(similarity_matrix)

		# graph, where edges are weights
		self.G = jaccard_graph
		self.M = similarity_matrix

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

	def get_metacell_sizes(self):
		return np.sum(self.assignments_bool, axis=0)

	def prune_small_metacells(self, thres:int=1):
		metacell_sizes = self.get_metacell_sizes()
		self.centers = self.centers[metacell_sizes >= thres]

	def get_metacell_graph(self):
		"""New connectivity graph for meta-cells"""
		return (self.assignments_bool.T @ self.G @ self.assignments_bool > 0).astype(float)

	def k_medoids(self, min_size, max_iter, init_centers=None, k=50, epsilon=1., max_centers=200):

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
