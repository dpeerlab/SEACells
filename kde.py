import numpy as np
import math
from scipy.special import gamma

from sklearn.neighbors import NearestNeighbors, kneighbors_graph

###############################
# Kernel functions (1D)
###############################

def K_gaussian(z):
	const = 1./(2. * math.pi)
	exponent = np.exp(- (z**2) / 2.)
	return const * exponent

def K_triangular(z):
	return np.maximum(0, 1 - np.abs(z))

################################
# Implementation of KDE class
################################

class KDE:
	"""Adaptive bandwidth KDE

	Attributes:
	---
	n_neighbors (int)
	verbose (bool)
	"""
	def __init__(self, n_neighbors:int=1, verbose:bool=True):
		self.n_neighbors = n_neighbors
		self.verbose = verbose

	def _fit(self, X):
		"""Given data X, compute KDE
		"""
		# data shape
		self.n, self.d = X.shape

		# save data
		self.X = X

		# construct nearest neighbors object
		self.nn = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)

		# compute bandwidths
		self._bandwidths()

	def _bandwidths(self):
		"""
		Compute bandwidth at each point in data set
		"""
		# query nearest neighbors for new points
		indices = self.nn.kneighbors(n_neighbors=self.n_neighbors, return_distance=False)

		H = np.zeros((self.n, self.d))

		# compute mean distance to nearest neighbors in each dimension
		for ix in range(self.n):
			for nbr in indices[ix,:]:
				H[ix,:] += np.abs(self.X[ix,:] - self.X[nbr,:]) / self.n_neighbors

		# store volume and bandwidth
		self.H_ = H

		# work in log space
		self.V_ = np.sum(np.log(H), axis=1 )# may be overflow, should convert to log

		# neighbor indices
		self.indices_ = indices

	def _predict(self, X_new):
		"""Predict density at a single point
		"""
		if len(X_new.shape) < 2:
			X_new = X_new.reshape(-1, self.d)

		n_new, d = X_new.shape

		# get stored values
		indices = self.indices_
		H = self.H_
		V = self.V_

		# initialize densities
		densities = np.zeros(n_new)

		# full loopy sum for now
		for ix in range(n_new):
			denom = np.log(self.n_neighbors) + V[ix]
			bw = H[ix]
			for j in range(self.n_neighbors):
				nbr = indices[ix][j]
				z = np.abs(X_new[ix,:] - self.X[nbr,:]) / bw
				densities[ix] += np.prod(K_gaussian(z))
			densities[ix] = np.log(densities[ix]) - denom

		return densities


	def fit(self, X):
		self._fit(X)
		return self

	def predict(self, X):
		densities = self._predict(X)
		return densities

	def local_maxima(self, k):
		"""Return local maxima"""
		densities = self.predict(self.X)


		#kneighbors = self.nn.radius_neighbors(self.X, radius=return_distance=False)
		ismax = np.zeros(self.n, dtype=bool)
		for ix in range(self.n):
			#r = np.exp((self.V_[ix] + gamma(self.d/2 + 1 - self.d / 2 * np.log(math.pi)))/self.d)
			#print(r) #* self.n**-0.2 #* self.n**(-1/5.)
			nbrs = self.nn.kneighbors(self.X[ix,:].reshape(1,-1), 
				n_neighbors=k, return_distance=False)[0]
			ismax[ix] = np.all(densities[nbrs] <= densities[ix])

		return np.arange(self.n)[ismax]

	def cluster(self):
		pass


if __name__=="__main__":
	model = KDE(5)
	X = np.random.randn(30,3)
	model.fit(X)
	print(model.predict(X))
