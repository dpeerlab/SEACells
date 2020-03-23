from collections import Counter
import numpy as np
from sklearn.datasets import make_swiss_roll, make_s_curve

class SwissRoll:

	def __init__(self, n_centers:int, verbose:bool=False):
		self.n_centers = n_centers
		self.verbose = verbose
		self.manifold = make_swiss_roll

	def sample(self, n:int, gamma_shape:float=1.0, gamma_scale:float=1.0):

		if self.verbose:
			print("Sampling a total of %d points..." % n)

		# total number of points
		self.n = n

		# location of centers, coordinate of centers
		self.centers, self.t = self.manifold(self.n_centers)

		# original assignments of full points
		self.assignments = np.random.choice(range(self.n_centers), self.n)

		# centers of full points
		self.full_centers = self.centers[self.assignments]
		self.full_t = self.t[self.assignments]

		# actual coordinates
		self.variances = np.random.gamma(gamma_shape, scale=gamma_scale, size=self.n)
		self.full_coordinates = np.zeros((self.n, 3))

		for i in range(self.n):
			mean = self.full_centers[i,:]
			cov = np.eye(3) * self.variances[i]
			self.full_coordinates[i,:] = np.random.multivariate_normal(mean, cov)

		# compute sizes of each center
		sizes_counter = Counter(self.assignments)
		self.sizes = np.array([sizes_counter[idx] for idx in range(self.n_centers)])

class SCurve:

	def __init__(self, n_centers:int, verbose:bool=False):
		self.n_centers = n_centers
		self.verbose = verbose
		self.manifold = make_s_curve

	def sample(self, n:int, gamma_shape:float=1.0, gamma_scale:float=1.0):

		if self.verbose:
			print("Sampling a total of %d points..." % n)

		# total number of points
		self.n = n

		# location of centers, coordinate of centers
		self.centers, self.t = self.manifold(self.n_centers)

		# original assignments of full points
		self.assignments = np.random.choice(range(self.n_centers), self.n)

		# centers of full points
		self.full_centers = self.centers[self.assignments]
		self.full_t = self.t[self.assignments]

		# actual coordinates
		self.variances = np.random.gamma(gamma_shape, scale=gamma_scale, size=self.n)
		self.full_coordinates = np.zeros((self.n, 3))

		for i in range(self.n):
			mean = self.full_centers[i,:]
			cov = np.eye(3) * self.variances[i]
			self.full_coordinates[i,:] = np.random.multivariate_normal(mean, cov)

		# compute sizes of each center
		sizes_counter = Counter(self.assignments)
		self.sizes = np.array([sizes_counter[idx] for idx in range(self.n_centers)])