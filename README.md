# metacells
working implementation of metacell tool for developing fine-grained, homogeneous clusters of single cell data

# dependencies
```
numpy
scipy
tqdm
```
# Usage
## initialisation
Initialize a model with an (n x d) matrix representing the PCA embedding of your data. If using scanpy, you can run
```
import mavs
model = mavs.mavs(ad.obsm["X_pca"])
```
## similarity matrix construction
The Jaccard similarity matrix as follows:
```
k = 15 # degree of kNN graph
model.initialize_kernel_jaccard_parallel(k)
```
## compute transition probabilities
Before clustering, the Markov transition matrix needs to be computed 
```
model.compute_transition_probabilities()
```
## clustering
The following will produce c clusters:
As a warning, this can take a while for data sets with more than 15K cells.
```
c = 300 # number of metacells
model.cluster(c)
```
## size distribution
sometimes it is useful to look at size distribution of each metacell to make sure nothing weird is happening. you can compute metacell sizes as follows:
```
metacell_sizes = model.get_soft_metacell_sizes()
```
The result will be a length c array, where c is the number of metacell centers.
## accessing selected centers
It is helpful to visualise selected centers on a low-dim embedding to make sure that all of your cell types are covered. The indices of selected centers are stored as a NumPy array in the attribute ```model.centers```. To plot the UMAP embedding, you can run
```
import matplotlib.pyplot as plt
plt.scatter(ad.obsm["X_umap"][model.centers,0], ad.obsm["X_umap"][model.centers,1])
```
for example.
## computing metacell gene/peak expressions
To get average gene expression corresponding to each metacell. The result will be a (c x d) matrix, where c is the number of metacells and d is the number of genes (or peaks).
```
average_expression = model.get_metacell_coordinates(ad.X)
```
