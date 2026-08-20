## SEACells:

**S**ingle-c**E**ll **A**ggregation for High Resolution **Cell S**tates

SEACells identifies **metacells**: groups of cells in the same biological state, found by
archetypal analysis on a nearest-neighbor kernel built from a low-dimensional embedding
(`X_pca` for scRNA-seq, `X_svd` for scATAC-seq). This denoises the data while preserving
heterogeneity, giving a high-resolution set of cell states for downstream analysis. See the
[paper](https://www.nature.com/articles/s41587-023-01716-9) for the method.

#### Installation

Uses [**uv**](https://docs.astral.sh/uv/). Install uv once with
`curl -LsSf https://astral.sh/uv/install.sh | sh`, then:

```
git clone https://github.com/dpeerlab/SEACells.git && cd SEACells
uv sync                 # CPU — works anywhere
uv sync --extra gpu     # + RAPIDS/CuPy/FAISS (NVIDIA GPU, CUDA 13, Linux/x86_64)
uv sync --extra dev     # + linting/pre-commit hooks
```

This builds a `.venv` with SEACells installed editable. Run code with
`uv run python ...` or `source .venv/bin/activate`. The GPU wheels come from the
NVIDIA pip index (preconfigured in `pyproject.toml`); validated on A100 80GB.

#### Running SEACells (CPU & GPU)

The core API is unchanged. A minimal run:

```python
import SEACells

# ad: AnnData with a low-dim embedding in ad.obsm ('X_pca' for RNA, 'X_svd' for ATAC)
model = SEACells.core.SEACells(
    ad,
    build_kernel_on='X_pca',   # 'X_svd' for scATAC
    n_SEACells=90,             # number of metacells (heuristic: ~1 per 75 cells)
)
model.construct_kernel_matrix()
model.fit(min_iter=10, max_iter=100)   # converges in ~15-50 iterations

# metacell assignments are written to ad.obs['SEACell']; aggregate raw counts:
meta_ad = SEACells.core.summarize_by_SEACell(ad, SEACells_label='SEACell', summarize_layer='raw')
```

**GPU acceleration (optimized).** Pass `use_gpu=True, use_unified=True` to run the
end-to-end GPU implementation (`SEACells.model.SEACellsModel`) — everything else is
identical:

```python
model = SEACells.core.SEACells(
    ad, build_kernel_on='X_pca', n_SEACells=90,
    use_gpu=True,        # run on GPU (needs cupy + cuML / RAPIDS)
    use_unified=True,    # use the optimized unified backend
)
model.construct_kernel_matrix()
model.fit(min_iter=10, max_iter=100)
```

It keeps the kernel and weight matrices resident on the GPU, uses exact GPU kNN
(cuML) and a memory-scalable reconstruction error, and scales to ~100k cells in a few GB
(a full 100k-cell fit runs in ~10 min on one A100; see
[`docs/gpu_speed_and_scale.md`](docs/gpu_speed_and_scale.md)).
`use_unified=True` also works with `use_gpu=False` (an optimized, single-source CPU path).

**Backward compatible.** `use_unified` defaults to `False`, so existing code is unchanged:
the default CPU path (`use_gpu=False`) and the legacy `use_gpu=True` / `use_sparse=True`
backends all behave exactly as before. `use_unified` is strictly opt-in.

#### Usage

1. <b>ATAC preprocessing</b>:
   `notebooks/ArchR` folder contains the preprocessing scripts and notebooks including peak calling using NFR fragments. See notebook [here](https://github.com/dpeerlab/SEACells/blob/main/notebooks/ArchR/ArchR-preprocessing.ipynb) to get started. A version of ArchR that supports NFR peak calling is available [here](https://github.com/dpeerlab/ArchR).

2. <b>Computing SEACells</b>:
   A tutorial on SEACells usage and results visualization for single cell data can be found in the [SEACell computation notebook] (https://github.com/dpeerlab/SEACells/blob/main/notebooks/SEACell_computation.ipynb).

3. <b>Gene regulatory toolkit</b>:
   Peak gene correlations, gene scores and gene accessibility scores can be computed using the [ATAC analysis notebook] (https://github.com/dpeerlab/SEACells/blob/main/notebooks/SEACell_ATAC_analysis.ipynb).

4. <b>TF activity inference</b>:
   TF activities along differenitation trajectories can be computed using the [TF activity notebook] (https://github.com/dpeerlab/SEACells/blob/main/notebooks/SEACell_tf_activity.ipynb).

5. <b>Large-scale data integration using SEACells </b>:
   Details are avaiable in the [COVID integration notebook] (https://github.com/dpeerlab/SEACells/blob/main/notebooks/SEACell_COVID_integration.ipynb)

6. <b>Cross-modality integration </b>:
   Integration between scRNA and scATAC can be performed following the [Integration notebook](https://github.com/dpeerlab/SEACells/blob/main/notebooks/SEACell_domain_adapt.ipynb)

#### Citations

SEACells manuscript is available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.04.02.486748v1). If you use SEACells for your work, please cite our paper.

```
@article {Persad2022.04.02.486748,
	author = {Persad, Sitara and Choo, Zi-Ning and Dien, Christine and Masilionis, Ignas and Chalign{\'e}, Ronan and Nawy, Tal and Brown, Chrysothemis C and Pe{\textquoteright}er, Itsik and Setty, Manu and Pe{\textquoteright}er, Dana},
	title = {SEACells: Inference of transcriptional and epigenomic cellular states from single-cell genomics data},
	elocation-id = {2022.04.02.486748},
	year = {2022},
	doi = {10.1101/2022.04.02.486748},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/04/03/2022.04.02.486748},
	eprint = {https://www.biorxiv.org/content/early/2022/04/03/2022.04.02.486748.full.pdf},
	journal = {bioRxiv}
}

```

---

## Release Notes
