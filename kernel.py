"""Implements a bunch of different kernels (not particularly efficiently)

Vanilla RBF
Adaptive bandwidth RBF
Diffusion kernel
	+ approximation with top DC's
	from full RBF kernel
Adaptive bandwidth RBF + nystrom approximation
Jaccard kernel
"""
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix, csr_matrix, vstack

# for parallelizing stuff
from multiprocessing import cpu_count, Pool
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

# optimization
import time

# get number of cores for multiprocessing
NUM_CORES = cpu_count()

def rff_vanilla(X, gamma, n_components):

	return RBFSampler(gamma=gamma, n_components=n_components, random_state=1).fit_transform(X)


def rff(X, n_neighbors, n_components):
	"""Random fourier features for adaptive bandwidth kernel"""
	# RANDOM FOURIER FEATURES

	# should probably parallelize this
	kng = kneighbors_graph(X, n_neighbors=n_neighbors, mode="distance")
	kn_distance = np.max(kng, axis=1).toarray().reshape(-1,1)
	#print(kn_distance)

	knscale = (1./kn_distance)**2
	#print(knscale)

	nc = n_components

	ff_scale = np.zeros((X.shape[0], nc))
	for ix, bw in tqdm(enumerate(knscale)):
	    ff_model = RBFSampler(gamma=bw, n_components=nc, random_state=1)
	    ff_scale[ix,:] = ff_model.fit_transform(X[ix,:].reshape(1,-1)).ravel()
	    
	return ff_scale

def rbf(X, sigma):
	"""Radial basis function kernel
	"""
	dist_mtx = cdist(X, X)
	square_distances = dist_mtx ** 2
	scaled_square_distances = square_distances / sigma

	return np.exp(-scaled_square_distances)

def adaptive_rbf(X, k):
	"""Adaptive bandwidth kernel RBF
	"""
	dist_mtx = cdist(X, X)
	square_distances = dist_mtx ** 2

	kneighbor_distances = np.max(kneighbors_graph(X, n_neighbors=k, mode="distance", include_self=False).toarray(), 
		axis=1, keepdims=True)

	scaling_mtx = kneighbor_distances @ kneighbor_distances.T 
	scaled_square_distances = square_distances / scaling_mtx

	return np.exp(-scaled_square_distances)

def adaptive_rbf_nystrom(X, k, n_landmarks):
	"""Adaptive bandwidth RBF kernel, with Nystrom approximation
	"""
	print("Computing full kernel")
	K = adaptive_rbf(X, k)

	print("Sampling landmarks")

	# randomly pick a few landmark points
	landmarks = np.random.choice(range(X.shape[0]), n_landmarks, replace=False)

	# invert landmark matrix
	Kuu_inverse = np.linalg.inv(K[:,landmarks][landmarks,:])

	w, v = np.linalg.eigh(Kuu_inverse)

	# include
	include = w > 0
	w = w[include]
	v = v[:,include]

	c = v @ np.diag(w) @ v.T

	K_nystrom = K[:,landmarks] @ c @ K[landmarks,:]

	return K_nystrom

def fancy_initialization(X, k, scale, error=0.1):
	"""Fancy column selection for nystrom"""

	# sample according to leverage scores?
	u, s, v = np.linalg.svd(X)

	# compute norms
	norms = np.linalg.norm(u, axis=1)**2

	# compute probabilities
	p = norms / np.sum(norms)

	# sample from p; sel is selected column index
	sel = np.random.choice(range(X.shape[0]), int(k / error), replace=True, p=p)

	# sampled distances
	X_landmark = X[sel,:]
	scale_landmark = scale[sel,:]
	scale_mtx = scale_landmark @ scale_landmark.T
	landmark_square_distances = cdist(X_landmark, X_landmark)**2
	K_landmark = np.exp(-landmark_square_distances / scale_mtx)

	# rescale columns
	rescale_mtx = 1.#len(sel) * np.sqrt(p[sel].reshape(-1,1) @ p[sel].reshape(1,-1))
	K_landmark_rescale = K_landmark / rescale_mtx

	# get SVD
	s, u = np.linalg.eig(K_landmark_rescale)

	return u[:,:k].real @ (np.diag(s[:k].real**-0.5)), sel


def ARD(X):

	n, d = X.shape

	out = np.ones((n, n))
	for dim in range(d):
		dim_std = np.std(X[:, dim])
		optimal_bandwidth = (dim_std * 1.06 * n**-0.2)**-1
		out = out * rbf_kernel(X[:,dim].reshape(-1,1), gamma=optimal_bandwidth)

	return out


def ARD_helper(X, bw):
	n, d = X.shape

	out = np.ones((n, n))
	for dim in range(d):
		optimal_bandwidth = bw[dim]
		out = out * rbf_kernel(X[:,dim].reshape(-1,1), gamma=optimal_bandwidth)

	return out



def ARD_nystrom(X, n_landmarks:int=50):

	n, d = X.shape

	# pick landmarks
	landmarks = np.random.choice(range(n), n_landmarks, replace=False)

	# calculate bandwidths
	bw = (np.std(X, axis=0) * (n ** -0.2) * 1.06)**-1

	# Kuu
	Kuu = ARD_helper(X[landmarks,:], bw)

	# get square root of inverse
	u, s, v = np.linalg.svd(Kuu)

	Kuu_sqrtinv = u @ np.diag(s**-0.5)

	# Kuf
	Kuf = np.ones((n, n_landmarks))

	for dim in range(d):
		optimal_bandwidth = bw[dim]
		Kuf = Kuf * rbf_kernel(X[:, dim].reshape(-1,1), X[landmarks,dim].reshape(-1,1), gamma=optimal_bandwidth)

	return Kuf @ Kuu_sqrtinv



def adaptive_rbf_nystrom_fast(X, k, n_landmarks, error=0.1):

	scale = np.max(kneighbors_graph(X, n_neighbors=k, mode="distance", include_self=False).toarray(), 
		axis=1, keepdims=True)

	print("Sampling landmarks")

	Kuu, landmarks = fancy_initialization(X, n_landmarks, scale, error=error)


	# randomly pick a few landmark points
	#landmarks = np.random.choice(range(X.shape[0]), n_landmarks, replace=False)

	# landmark coordinates
	# X_landmark = X[landmarks,:]
	# scale_landmark = scale[landmarks,:] @ scale[landmarks,:].T

	# #print(np.mean(1/scale))

	# # sampled distances
	# landmark_square_distances = cdist(X_landmark, X_landmark)**2
	# K_landmark = np.exp(-landmark_square_distances / scale_landmark)

	# # matrix square root
	# # invert using cholesky factor
	# c = np.linalg.inv(K_landmark)
	# w, v = np.linalg.eigh(c)

	# # include
	# include = w > 0
	# w = w[include]
	# v = v[:,include]
	# # print(w)
	# # print(np.linalg.inv(K_landmark))
	# # print(v @ np.diag(w) @ v.T)

	# # square root
	# K_landmark_sqrt = v @ np.diag(w**0.5)

	scale_uf = scale @ scale[landmarks,:].T
	K_uf = np.exp(-cdist(X, X[landmarks,:])**2 / scale_uf) @ Kuu#@ K_landmark_sqrt
	return K_uf, scale.ravel()

def adaptive_rbf_sparse(X, k):
	dist_mtx = cdist(X, X)
	square_distances = dist_mtx ** 2

	kneighbor_distances = np.max(kneighbors_graph(X, n_neighbors=k, mode="distance", include_self=False).toarray(), 
		axis=1, keepdims=True)

	scaling_mtx = kneighbor_distances @ kneighbor_distances.T 
	scaled_square_distances = square_distances / scaling_mtx

	# compute mask (get rid of anything more than 3 stds away)
	mask = (scaling_mtx**0.5 * 2 - dist_mtx > 0).astype(float)

	return np.exp(-scaled_square_distances) * mask

def adaptive_rbf_helper(X, ka_distances, idx, threshold_std = 1.5):
	"""Helper function for adaptive bandwidth RBF
	"""
	# get number of points
	n = int(X.shape[0])

	# coordinate of current point under consideration
	pt = X[idx,:].reshape(1,-1)

	# get distance from point to all other points
	distances = cdist(X, pt).ravel()

	# get threshold distance for each point
	threshold_distance = ka_distances[idx] * ka_distances

	# mask distances with threshold
	mask = (distances < threshold_std * np.sqrt(threshold_distance)).astype(int)

	# create sparse matrix
	log_row = coo_matrix(mask * np.exp(-(distances**2)/threshold_distance), shape=(1,n))
	
	# exponentiate nonzero entries
	return log_row

def adaptive_rbf_parallel(X, k):
	"""Parallel construction of adaptive bandwidth RBF kernel

	Sparse (all distances 3 std away become zero)
	"""
	# number of points
	n = X.shape[0]

	# compute kNN graph
	kng = kneighbors_graph(X, n_neighbors=k, mode="distance", include_self=False)

	# get distance to kth nearest neighbo
	kn_distances = np.max(kng, axis=1).toarray().squeeze(-1)

	# compute rows in parallel (need: index of point, point coordinates, ka neighbor distances)
	with Parallel(n_jobs=NUM_CORES, backend="threading") as parallel:
		mtx_rows = parallel(delayed(adaptive_rbf_helper)(X, kn_distances, i) for i in tqdm(range(n)))

	# stack rows
	return vstack(mtx_rows).tocsr()


def diffusion(X, k, t):
	"""Diffusion kernel
	"""
	K = adaptive_rbf(X, k)

	# row normalize
	K_normalized = K / K.sum(axis=1, keepdims=True)

	# compute eigenvalues
	w, v = np.linalg.eig(K_normalized)

	return v.T @ np.diag(w)**t @ v

def kmpp(Y, k):
	"""K means ++"""
	centers = np.zeros(k, dtype=int)

	# randomly pick first point
	ix = 0
	p = Y[ix,:]
	centers[0] = ix
	distances = cdist(Y, p.reshape(1, -1)).ravel()
	for i in range(k-1):
		ix = np.argmax(distances)
		centers[i+1] = ix
		p = Y[ix,:]
		new_distances = cdist(Y, p.reshape(1,-1)).ravel()
		#stacked_distances = np.vstack([distances, new_distances])
		#distances = np.min(stacked_distances, axis=0)
		distances += new_distances
		distances[centers] = 0
	return centers


def jaccard(X, k):
	"""Jaccard similarity kernel
	"""
	pass