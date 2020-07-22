from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
import numpy as np

from collections import deque

class rsp:
    """Recursive spectral partitioning"""

    def __init__(self, gamma=1., verbose=True):

        # resolution parameter
        self.gamma = gamma
        self.assignments = None
        # verbosity
        self.verbose = verbose

    def _modularity(self, G, partition):
        """G: a CSR matrix
        partition: boolean array, numpy array"""

        #G = (G > 0).astype(int)

        k = G.sum(axis=1)
        m = np.sum(k)

        return 1./(2*m) * np.trace(partition.T @ G @ partition - partition.T @ k @ k.T @ partition / (2*m))

    def _cpm(self, G, partition, gamma):
        """Constant Potts model"""

        # edges within partition
        p1 = partition[:,0].astype(bool)
        gp1 = (G[p1,:][:,p1] > 0).sum()
        p2 = partition[:,1].astype(bool)
        gp2 = (G[p2,:][:,p2] > 0).sum()

        term1 = gp1 + gp2
        print(term1)

        cluster_sizes = partition.sum(axis=0)
        print(cluster_sizes)
        term2 = np.sum(cluster_sizes * (cluster_sizes-1) / 2)

        print(term2*gamma)

        # try mean edge weight
        #mew = np.mean(G.data) #/ (G.shape[0] * G.shape[1])
        #print(mew)

        return term1 - gamma * term2 #* mew

    def _split_graph(self, original_graph, selection):

        # select subgraph
        G = original_graph[selection,:][:,selection]

        # get Laplacian
        rows = G.shape[0]
        cols = G.shape[1]
        diag = csr_matrix((np.squeeze(np.array(G.sum(axis=1))), (np.arange(rows), np.arange(cols))), 
            shape=G.shape)
        L = diag - G

        # get first two eigenvectors
        w, v = eigs(L.astype(float), k=2, which="SM")
        #print(w)

        # get fiedler vector sign
        fiedler = (v[:, 1] > 0).astype(int)

        # return partition
        partition = np.zeros((rows, 2))
        partition[np.arange(rows), fiedler] = 1

        return partition.astype(int)

    def fit(self, G, min_size=50.):
        """Run method"""
        n = G.shape[0]
        self.n=n

        # initialize cluster assignments
        self.assignments = np.zeros((n, 1))

        # for now keep track of to-dos in FIFO queue
        sel = np.ones(n, dtype=int)
        fifo = deque([(sel, 0)])

        while fifo:
            new_selection, layer = fifo.popleft()
            new_selection = new_selection.astype(bool)

            if layer >= self.assignments.shape[1]:
                self.assignments = np.hstack([self.assignments, np.zeros((n, 1))])

            # check to make sure selection is not too small
            if np.sum(new_selection) <= min_size:
                continue

            # define new partition
            new_partition = self._split_graph(G, new_selection)

            # check modularity
            subgraph = G[new_selection,:][:,new_selection]
            # Modularity of null model is 0.25 (so should only keep partitions that improve modularity)
            original_modularity = self._modularity(subgraph, np.ones((subgraph.shape[0], 1)))
            modularity = self._modularity(subgraph, new_partition)

            if self.verbose:
                print("Original modularity: ", original_modularity)
                print("Modularity: ", modularity)

            if modularity < 0:
                continue

            # size check
            if np.any(np.sum(new_partition, axis=0) < min_size):
                continue

            #print(new_partition)

            # update partition
            partition_index = np.where(new_selection)
            partition_0 = np.zeros(n)
            partition_0[partition_index[0]] = new_partition[:,0]
            partition_1 = np.zeros(n)
            partition_1[partition_index[0]] = new_partition[:,1]

            # add to queue
            fifo.append((partition_0, layer + 1))
            fifo.append((partition_1, layer + 1))

            # update assignments
            self.assignments[partition_index[0], layer] = new_partition[:,0]
            #print(self.assignments)
            #if self.verbose:
                #print("Current assignment: ")
                #print(self.get_final_assignments())
        self.compress_assignments()
        self.boolean_assgts()

    def get_final_assignments(self):
        binary_strings = ["".join(list(row.astype(int).astype(str))) for row in self.assignments]
        return [int(s[::-1], 2) for s in binary_strings]

    def compress_assignments(self):
        assgts = self.get_final_assignments()
        unique = set(assgts)
        mapping = {a:ix for ix,a in enumerate(unique)}
        self.compressed = np.array([mapping[a] for a in assgts], dtype=int)
        #return self.compressed

    def boolean_assgts(self):
        assgts = self.compressed
        out = np.zeros((len(self.compressed), len(set(self.compressed))))
        out[np.arange(len(self.compressed)), assgts] = 1
        self.bools = out
        #return out

    def get_metacell_sizes(self):
        return self.bools.sum(axis=0)

    def get_coordinates(self, coordinates):
        """Aggregates reads"""
        bools = self.bools / self.bools.sum(axis=0, keepdims=True)
        return (bools.T @ coordinates) #/ bools.T.sum(axis=1, keepdims=True)

    def get_metacell_labels(self, labels, exponent=1.):
        """Given labels of original cells, transfer labels
        Exponent: optional softmax tempering
        """
        # get onehot encoding of labels
        unique_labels = set(labels)
        label2idx = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label = {idx:label for label,idx in label2idx.items()}

        # print(label2idx)

        # onehot labels
        onehot_labels = np.zeros((self.n, len(unique_labels)))
        label_indices = np.array([label2idx[label] for label in labels])
        onehot_labels[np.arange(self.n), label_indices] = 1.

        # print(onehot_labels)

        # get soft label assignments
        W = self.bools
        W = W / W.sum(axis=0, keepdims=True)
        metacell_labels = W.T @ onehot_labels

        # print(metacell_labels)

        # get hard labels
        metacell_hard_labels = np.argmax(metacell_labels, axis=1)

        # print(metacell_hard_labels)

        # get actual word labels
        return [idx2label[idx] for idx in metacell_hard_labels]


def main():
    G = [[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    G = csr_matrix(G)

    model = rsp(1.0, verbose=True)
    model.fit(G)

    print("FINAL ASSIGNMENTS")
    print(model.get_final_assignments())

if __name__=="__main__":
    main()








