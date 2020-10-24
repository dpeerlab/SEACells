# metacells
working implementation of metacells

# dependencies

```
numpy
scipy
sklearn
tqdm
```

# Usage

For a demo, see the notebook ```guttube_aa.ipynb```.

The gut endoderm single cell RNA-seq data used to prepare the demo is available on S3:

```s3://dp-lab-home/znchoo/gut_endoderm/data/E85_endoderm_processed.h5ad```

# Inputs

The input to the the clustering alg is a similarity matrix, which requires a low-dimensional representation of the data to compute. Usually I use PCA on HVGs and truncated SVD on either peaks or ArchR bins for RNA and ATAC, respectively.

## Building similarity graph

The following code assumes that you have an ```AnnData``` named ```ad``` and the low-dim representation mentioned above stored in ```obsm``` with key ```"X_pca"```.

```
import build_graph

graph_model = build_graph.MetacellGraph(ad.obsm["X_pca"], verbose=True)
G = graph_model.rbf()
```

## Finding metacells

To find metacells you just need to input the graph ```G``` from above, plus the number of metacells that you want to find. I find that a good rule of thumb is the number of cells in your data set divided by 30, or so.

```
reload(metacells)

# set number of metacells
N_METACELLS = 700
metacell_model = metacells.Metacells(n_metacells=N_METACELLS)

# use the graph G from above
metacell_model.fit(G);
```

## Downstream

### Size distribution

To get an array of the sizes of the metacells, run the following

```
sizes = metacell_model.get_sizes()
```

### Coordinates

To get metacell-level UMI counts:

```
metacell_coords = metacell_model.get_coordinates(ad.X)
```

The above assumes that ```X``` represents the raw (integer-valued) counts of each gene per cell in your AnnData.

### Get metacell assignments

For each single cell in your data set, get the (numerical) index of the corresponding metacell

```
assgts = metacell_model.get_assignments()
```

### Get centers

Get the index (integer) of the closest cell in the data to the average of each metacell.

```
cx = metacell_model.get_centers()
```
