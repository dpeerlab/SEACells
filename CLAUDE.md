# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

SEACells (Single-cEll Aggregation for High Resolution Cell States) is a Python package that infers metacells from single-cell genomics data (scRNA, scATAC, multiome) using kernel archetypal analysis. The library is consumed primarily through the example Jupyter notebooks in `notebooks/`; there is no CLI or test suite.

## Common commands

Developer install (from the repo root):

```
pip install -e ".[dev]"
pre-commit install
```

Lint / format (no tests are defined in this repo):

```
pre-commit run --all-files     # runs black, isort, ruff, blacken-docs, prettier
ruff check SEACells/           # ruff alone
```

Build a conda env from the pinned spec (note: `environment.yaml` pins `python=3.5` but `setup.py` requires `>=3.8` — the README's option (4) using a fresh `python=3.8` conda env plus `pip install -r requirements.txt` is the path that actually works):

```
conda create --name seacells -c conda-forge -c bioconda cython python=3.8
conda activate seacells
pip install -r requirements.txt
pip install -e .
```

Notebooks are the primary way to exercise the code:

```
jupyter lab notebooks/SEACell_computation.ipynb
```

## Architecture

The public API is re-exported from `SEACells/__init__.py` (`core`, `preprocess`, `utils`, `plot`). The rest of the package is imported on demand by callers.

### Core compute path

`SEACells.core.SEACells(ad, build_kernel_on, n_SEACells, ..., use_gpu, use_sparse)` is a **factory** — not a class. It dispatches to one of three backend implementations based on flags:

- `use_sparse=True` → `cpu.SEACellsCPU` (sparse CSR kernel, scipy.sparse + sklearn)
- `use_gpu=True` → `gpu.SEACellsGPU` (CuPy-based; only import on demand)
- default → `cpu_dense.SEACellsCPUDense` (dense numpy kernel)

All three backends share the same constructor signature and expose `.fit()`. They each implement: kernel construction (via `build_graph.py`), waypoint-based archetype initialization (uses `palantir` diffusion components), and a Franke–Wolfe style optimization loop. When editing the algorithm, changes generally need to be mirrored across all three backends — they are intentionally parallel implementations, not a shared base class. `cpu.py` and `cpu_dense.py` differ mainly in dense-vs-sparse linear algebra; `gpu.py` mirrors `cpu_dense.py` on CuPy.

The Frank–Wolfe inner loop in `cpu.py:_updateA / _updateB` has been rewritten to densify the assignment / archetype matrices for the inner loop and to track `t1 @ A` / `K @ B` incrementally (the kernel `K` itself stays sparse; column slicing uses a one-time CSC view). `cpu_dense.py` and `gpu.py` still use the original recompute-the-gradient-from-scratch loop — if you mirror `cpu.py` algorithm changes there, port the inner-loop optimizations too. The function signatures and return types (`csr_matrix`) are unchanged, so `step`, `compute_RSS`, and `save_assignments` are unaffected.

The factory returns the model **unfitted** — callers must invoke `.fit()` themselves. SEACell assignments are written back to `ad.obs['SEACell']` in place.

### Dual-import pattern

Backend modules use a `try: from . import X / except ImportError: import X` pattern (see `core.py:8`, `cpu.py:11`, etc.). This lets the files run both as a package and as standalone scripts during development. Preserve this pattern when adding new intra-package imports.

### Aggregation helpers

`core.summarize_by_SEACell` (hard assignment, uses `ad.obs['SEACell']`) and `core.summarize_by_soft_SEACell` (soft assignment, uses an `A` matrix with `sparsify_assignments`) produce the metacell-level AnnData consumed by all downstream modules. Anything that operates on metacells expects the output of one of these.

Both functions are now single sparse matmuls — `summarize_by_SEACell` builds a metacell × cell indicator and returns `indicator @ data`; `summarize_by_soft_SEACell` returns `(A.T @ data) / totals` with a zero-guard for empty metacells. Hard-assignment metacell ordering follows first-occurrence (`pd.Series.unique()`); soft-assignment celltype tie-breaking uses `argmax` over a category-indicator matmul, which matches the prior `groupby(...).sort_values(...).iloc[0]` ordering when categories are alphabetic.

### Downstream modules

These are independent of the core optimization and operate on metacell AnnData objects:

- `genescores.py` — multiome workflow. `prepare_multiome_anndata` is the entry point that pairs ATAC + RNA AnnData by shared `SEACell` labels and produces matched metacell objects for peak–gene correlation and gene-score computation.
- `accessibility.py` — per-metacell open-peak calls.
- `tfactivity.py` — TF activity inference along trajectories.
- `domainadapt.py` — linear OT (`LinearOT`) for cross-modality / cross-batch alignment.
- `evaluate.py` — metacell quality metrics (`compactness`, `separation`, `compute_celltype_purity`). `core.summarize_by_SEACell` imports `evaluate` for purity computation, so avoid importing `core` from `evaluate` (would create a cycle).
- `plot.py` — plotting helpers built on scanpy/matplotlib.
- `Rscripts/` — auxiliary R scripts (`chromVAR.R`, `tanay.R`) shipped via `package_data` in `setup.py`; called out-of-process, not from Python.

### Data conventions

- `build_kernel_on` is an `ad.obsm` key — `'X_pca'` for scRNA, `'X_svd'` for scATAC. Callers are responsible for computing this beforehand (the notebooks show standard scanpy / ArchR pipelines).
- Raw counts are expected at `ad.raw.X` or `ad.layers['raw']`; `summarize_by_SEACell` aggregates from there and writes `meta_ad.layers['raw']`.
- `SEACells/data/sample_data.h5ad` is bundled for the tutorials.

## Tooling notes

- Formatting is enforced via pre-commit (black, isort, ruff with `--fix`, prettier, blacken-docs). Run `pre-commit run --all-files` before opening a PR; the hooks autofix most issues.
- `python_requires=">=3.8"` per `setup.py`. The conda `environment.yaml` is stale (pins 3.5) and the README's option (4) is the working install path.
- GPU backend depends on CuPy and is imported lazily — do not add a top-level `import cupy` anywhere.
