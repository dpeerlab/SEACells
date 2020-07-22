from functools import reduce
import numpy as np
from scipy.special import factorial, logsumexp
from scipy.sparse import coo_matrix, csr_matrix
import time

class gsdmm:

  def __init__(self, K: int, alpha: float, beta: float, verbose: bool=False):
    self.K = K
    self.alpha = alpha
    self.beta = beta
    self.verbose = verbose

  def initialize(self, X):
    """
    Args:
      X: sparse csr matrix of cells by genes
    """
    # store X
    self.X = X

    # convert to list of lists
    self.X_lil = X.tolil()

    # set number of cells and genes
    self.n_cells, self.n_genes = self.X.shape

    # get library sizes
    self.library_size = np.sum(self.X.toarray(), axis=1, keepdims=True).T

    # start with uniform cluster responsibilities
    pvals = np.ones(self.K) * (1./self.K)

    # sample cluster memberships from multivariate normal distribution
    self.z = np.random.multinomial(1, pvals, size=self.n_cells)

    if self.verbose:
      print("Initial cluster memberships shape: ", self.z.shape)

    # update m_z, n_z, and n_z_w
    self.m_z = np.ones((1, self.n_cells)) @ self.z
    self.n_z = self.library_size @ self.z
    self.n_z_w = self.z.T @ self.X

    if self.verbose:
      print("Shape of m_z: ", self.m_z.shape)
      print("Shape of n_z: ", self.n_z.shape)
      print("Shape of n_z_w: ", self.n_z_w.shape)

  def get_pvals(self, cell_index):
    """get complete conditional for cell index"""

    # get current cluster
    z = np.argmax(self.z[cell_index,:])

    # update m_z and n_z
    self.m_z[0, z] = self.m_z[0, z] - 1.
    self.n_z[0, z] = self.n_z[0, z] - self.library_size[0, cell_index]
    self.n_z_w[z,:] = self.n_z_w[z,:] - self.X[cell_index,:]

    # compute probability vector
    part_1_numerator_log = np.log(self.m_z[0,:] + self.alpha)
    part_1_denominator_log = np.log(self.n_cells - 1 + self.K*self.alpha)

    part_2_numerator_log = np.zeros(self.K)

    # iterate over words in the document  
    for gene_count, gene in zip(self.X_lil.data[cell_index], self.X_lil.rows[cell_index]):
      part_2_numerator_log += sum(map(lambda j: np.log(self.n_z_w[:, gene] + self.beta + j), range(gene_count)))
    
    """
    for idx, gene in enumerate(cell.col):
      for j in range(cell.data[idx]):
        # zero indexed so we don't have to subtract 1
        part_2_numerator_log += np.log(self.n_z_w[:, gene] + self.beta + j)
    """

    part_2_denominator_log = sum(map(lambda i: np.log(self.n_z[0, :] + self.n_genes*self.beta + i), range(self.library_size[0, cell_index])))
    """
    for i in range(self.library_size[0, cell_index]):
      # zero indexed so we don't have to subtract 1
      part_2_denominator_log += np.log(self.n_z[0, :] + self.n_genes*self.beta + i)
    """

    p_vals_log = part_1_numerator_log + part_2_numerator_log - part_1_denominator_log - part_2_denominator_log
    norm_constant_log = logsumexp(p_vals_log)
    p_vals_raw = np.exp(p_vals_log - norm_constant_log)
    p_vals_normalized = p_vals_raw / sum(p_vals_raw)

    if self.verbose:
      print("Part 1 numerator log: ", part_1_numerator_log)
      print("Part 1 denominator log: ", part_1_denominator_log)
      print("Part 2 numerator log: ", part_2_numerator_log)
      print("Part 2 denominator log: ", part_2_denominator_log)
      print("Log complete conditional: ", p_vals_log)
      print("Log normalization constant: ", norm_constant_log)
      print("Normalized complete conditional: ", p_vals_normalized)

    return p_vals_normalized

  def sample_new_cluster(self, cell_index, p_vals):
    # updates cluster assignment
    self.z[cell_index, :] = np.random.multinomial(1, p_vals, 1)
    if self.verbose:
      print("New cluster for cell %d: %d" % (cell_index, np.argmax(self.z[cell_index,:])))

  def update_cell(self, cell_index):
    # get updated cluster assignment
    z = np.argmax(self.z[cell_index,:])

    # update m_z, n_z, n_z_w
    self.m_z[0, z] = self.m_z[0, z] + 1.
    self.n_z[0, z] = self.n_z[0, z] + self.library_size[0, cell_index]
    self.n_z_w[z,:] = self.n_z_w[z,:] + self.X[cell_index,:]

  def step(self):
    for cell_index in range(self.n_cells):
      p_vals = self.get_pvals(cell_index)
      self.sample_new_cluster(cell_index, p_vals)
      self.update_cell(cell_index)

  def many_steps(self, max_iter=50):
    for n_iter in range(max_iter):
      start = time.time()
      print("Iteration %d of %d" % (n_iter, max_iter))
      self.step()
      end = time.time()
      print("Elapsed time: %.8f" % (end-start))

  def get_labels(self):
    return np.argmax(self.z, axis=1)

def main():
  # code for testing
  # make some fake cells
  n_cells = 10
  n_genes = 15
  p_vals = np.ones(n_genes) * (1./n_genes)
  test_X = np.random.multinomial(2, p_vals, n_cells)
  test_X_sparse = csr_matrix(test_X)

  # set params
  K = 3
  alpha = 0.1
  beta = 0.1

  # initialize model
  gsdmm_model = gsdmm(K, alpha, beta)
  gsdmm_model.initialize(test_X_sparse)
  gsdmm_model.many_steps()

if __name__ == "__main__":
  main()


