"""
Just regular Archetypal Analysis - no kernel, to be used with Random Fourier Features
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.710.1085&rep=rep1&type=pdf#page=8

To do: initialize B from greedily-selected columns from RRQR (seems unnecessary)
How to determine convergence?
"""

import numpy as np
from tqdm.notebook import tqdm
from scipy.spatial.distance import cdist

class ArchetypalAnalysis:
    """
    Fast kernel archetypal analysis.
    Finds archetypes and weights given precomputed kernel matrix.

    Attributes:
        k: number of components
        max_iter: maximum number of iterations for Frank-Wolfe update
        verbose: verbosity
    """

    def __init__(self, k:int, max_iter:int=50, verbose:bool=True):
        self.k = k
        self.max_iter = max_iter
        self.verbose = verbose

    def _updateA(self, X, A, B, Z):
        """
        Given archetype matrix B and feature matrix X, compute assignment matrix A

        Inputs:
            X: n*d feature matrix (dense)
            B: n*k matrix (dense)

        Returns:
            A: k*n matrix (dense)
        """
        k = self.k
        n, d  = X.shape

        # initialize matrix A (don't reinitialize?)
        A = np.zeros((k, n))
        A[0,:] = 1.

        t = 0 # current iteration (determine multiplicative update)

        # precompute some gradient terms
        #t2 = (X @ (Z.T)).T
        #t1 = t2 @ B
        t1 = Z @ Z.T
        t2 = Z @ X.T

        # update rows of A for given number of iterations
        while t < self.max_iter:

            # compute gradient (must convert matrix to ndarray)
            G = 2. * np.array(t1 @ A - t2)

            # get argmins
            amins = np.argmin(G, axis=0)

            # loop free implementation
            e = np.zeros((k,n))
            e[amins, np.arange(n)] = 1.

            A += 2. / (t + 2.) * (e - A)
            t += 1

        return A

    def _updateB(self, X, A, B):
        """Given assignment matrix A and feature matrix X, compute archetype matrix B

        Inputs:
            X: n*d kernel matrix (sparse)
            A: k*n matrix (dense)

        Returns:
            B: n*k matrix (dense)
        """
        k, n = A.shape
        d = X.shape[1]

        # initialize matrix B (don't re-initialize?)
        B = np.zeros((n,k))
        B[0, :] = 1.

        # keep track of error
        t = 0

        # precompute some terms
        t1 = A @ A.T # k x k
        t2 = X @ (X.T @ A.T) # shape: n x k

        # update rows of B for a given number of iterations
        while t < self.max_iter:

            # compute gradient (need to convert np.matrix to np.array)
            G = 2. * np.array((X @ (X.T @ B)) @ t1 - t2)

            # get all argmins
            amins = np.argmin(G, axis=0)

            e = np.zeros((n,k))
            e[amins, np.arange(k)] = 1.

            B += 2. / (t+2.) * (e - B)

            t += 1

        return B

    def _initialize_archetypes(self, X):
        """Fast greedy adaptive CSSP

        From https://arxiv.org/pdf/1312.6838.pdf

        Inputs:
            K (n*n) kernel matrix
        """
        n = X.shape[0]
        k = self.k

        # if self.verbose:
        #     print("Initializing residual matrix")

        # # precompute A.T * A
        # ATA = X @ X.T
        # #ATA = K

        # if self.verbose:
        #     print("Initializing f and g...")

        # #f = np.array((ATA.multiply(ATA)).sum(axis=0)).ravel()
        # f = np.array(ATA * ATA).sum(axis=0).ravel()
        # g = np.array(ATA.diagonal()).ravel()

        # d = np.zeros((k, n))
        # omega = np.zeros((k, n))

        # # keep track of selected indices
        # centers = np.zeros(k, dtype=int)

        # # sampling
        # for j in tqdm(range(k)):

        #     score = f/g
        #     p = np.argmax(score)

        #     # print residuals
        #     residual = np.sum(f)

        #     delta_term1 = ATA[:,p]#.toarray().squeeze()
        #     #print(delta_term1)
        #     delta_term2 = np.multiply(omega[:,p].reshape(-1,1), omega).sum(axis=0).squeeze()
        #     delta = delta_term1 - delta_term2

        #     # some weird rounding errors
        #     #delta[p] = np.max([0, delta[p]])

        #     o = delta / np.max([np.sqrt(delta[p]), 1e-6])
        #     omega_square_norm = np.linalg.norm(o)**2
        #     omega_hadamard = np.multiply(o, o)
        #     term1 = omega_square_norm * omega_hadamard

        #     # update f (term2)
        #     pl = np.zeros(n)
        #     for r in range(j):
        #         omega_r = omega[r,:]
        #         pl += np.dot(omega_r, o) * omega_r

        #     ATAo = (ATA @ o.reshape(-1,1)).ravel()
        #     term2 = np.multiply(o, ATAo - pl)

        #     # update f
        #     f += -2. * term2 + term1

        #     # update g
        #     g += omega_hadamard

        #     # store omega and delta
        #     d[j,:] = delta
        #     omega[j,:] = o

        #     # add index
        #     centers[j] = int(p)

        # create matrix B

        #centers = np.random.choice(range(n), k, replace=False)
        centers = np.zeros(k, dtype=int)

        # select initial point randomly
        new_point = np.random.choice(range(n), 1)
        centers[0] = new_point

        # initialize min distances
        distances = cdist(X, X[new_point, :].reshape(1,-1), metric="euclidean")

        # assign rest of points
        for ix in range(1, k):
            new_point = np.argmax(distances.ravel())
            centers[ix] = new_point
            #print(new_point)

            # get distance from all poitns to new points
            new_point_distances = cdist(X, X[new_point, :].reshape(1,-1), metric="euclidean")

            # update min distances
            combined_distances = np.hstack([distances, new_point_distances])

            distances = np.sum(combined_distances, axis=1, keepdims=True)

            # make sure that the distance of already added points is zero to prevent duplicates
            distances[centers,0] = 0

        B = np.zeros((n, k))
        B[centers, np.arange(k)] = 1.

        return B

    def _residuals(self, X, A, Z):
        """Use trace trick to compute residual squared error

        Actual formula for the error is
            E = ||X - XBA||^2
            = tr(X.T @ X 
                - 2 X.T * X * B * A 
                + A.T * B.T * X.T * X * B * A)

        (Trace distributes over sums and invariant to permutation)

        Inputs:
            K: n*d feature matrix (dense)
            A: k*n assignments matrix (dense)
            B: n*k archetype matrix (dense)

        Returns:
            E (float)
        """

        # term1 = np.trace(K)
        # term1 = K.shape[0]

        # first term is the diagonal of the kernel (aka square norm of X rows)
        term1 = np.sum(np.linalg.norm(X, axis=1)**2)
        term2 = np.trace((A @ X) @ (Z.T))
        term3 = np.trace((A.T @ (Z)) @ ((Z.T) @ A))

        return term1 - 2. * term2 + term3

    def _fit(self, X, n_iter:int):
        """Compute archetypes and loadings given kernel matrix K

        Input:
            X: feature matrix (n x d) dense np.ndarray
            n_iter: number of iterations

        Updates model to add B and A
        """
        # initialize B (update this to allow initialization from RRQR)
        n = X.shape[0]
        k = self.k

        B = self._initialize_archetypes(X)
        Z = B.T @ X

        A = np.eye(k, n)
        A[0,:] = 1.

        for it in range(n_iter):
            

            A = self._updateA(X, A, B, Z)
            B = self._updateB(X, A, B)

            Z = B.T @ X
            

            # compute error
            error = self._residuals(X, A, Z)

            print("Iteration %d of %d. Error: %.8f" % (it, n_iter, error))

        self.A_ = A
        self.B_ = B
        self.Z_ = B.T @ X

    def fit(self, X, n_iter:int=8):
        """Wrapper to fit model given kernel matrix and max number of iterations
        
        Inputs:
            X (np.ndarray): feature matrix (n x d)
            n_iter (int): number of optimization iterations

        Returns:
            self
        """
        self._fit(X, n_iter)
        return self

    def fit_transform(self, X, n_iter:int=10):
        """Fit model and return archetype assignments
        """
        self._fit(X, n_iter)
        return self.A_

    def transform(self, X_new):
        """ Given new set of points, compute assignment to previously discovered archetypes
        """
        try:
            B = self.B_
        except:
            raise AttributeError

        # get object shape
        n, d = X_new.shape

        # number of archetypes
        k = self.k

        A_guess = np.eye(k, n)
        A_guess[0,:] = 1.

        A_new = self._updateA(X_new, A_guess, B, self.Z_)
        return A_new

    def score(self, X_new):
        """Score quality of approximation of new points
        """
        A_new = self.transform(X_new)
        B = self.B_
        Z = self.Z_

        return self._residuals(X_new, A_new, Z)

    def get_archetypes(self):
        """Return k x n matrix of archetypes"""
        return self.Z_

    def get_centers(self):
        """Return closest point to each archetype"""
        return np.argmax(self.B_, axis=0)

    def get_assignments(self):
        """Return archetype assignments for each point (n x k)
        """
        return self.A_.T

    def get_coordinates(self, X):
        """Return cluster centers"""
        return np.array(self.A_ @ X) #/ self.A_.sum(axis=1, keepdims=True)

