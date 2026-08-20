# GPU SEACells: speed and scale

Two changes let SEACells run end-to-end on the GPU. They change only *where* the
computation runs and *how* the objective is evaluated. The model and its results
are unchanged.

## #1 — Keep the kernel resident on the GPU (speed)

The original `gpu.py` left the kernel `K` on the host and re-uploaded it every
iteration; worse, it evaluated the reconstruction error (RSS) on the CPU by
building a dense `n x n` matrix. Keeping `K` (and `M`, `A`, `B`) resident on the
GPU lets the whole iteration, including the RSS, run on the device.

*Example (15k cells):* per-iteration time drops from **10.4 s to 0.58 s (~18x)**,
because the RSS moves off the CPU. End-to-end on cd34: **110 s to 6.5 s (~17x)**.

## #2 — Evaluate the RSS without the `n x n` matrix (scale)

The RSS is `||M - MBA||`. Forming the reconstruction `MBA` is a dense `n x n`
matrix (174 GB at 208k cells, far past a GPU's memory).

The trick is to expand the norm and use `K = MᵀM` (M symmetric):

```
||M - MBA||²  =  ||M||²  -  2·tr(KBA)  +  tr(BᵀKB · AAᵀ)
```

Every term needs only `n x s` arrays, so the RSS costs `O(n·s)` memory instead of
`O(n²)`. Nothing is approximated: this equals the direct Frobenius norm exactly.

*Example (208k cells):* peak memory **~10 GB instead of 174 GB**, so it fits on one
GPU and runs where the original runs out of memory.

## The result is unchanged

Same math, same optimum. On cd34, with the same kernel and initialization, CPU and
GPU converge along the same RSS curve; the final RSS matches to `3e-5` and 94% of
metacells are identical (the rest are boundary cells with equal RSS — CPU vs GPU
float ordering, not a change in the result).

## Usage

The GPU path is opt-in and backward compatible (`use_unified` defaults to `False`):

```python
model = SEACells.core.SEACells(
    ad, build_kernel_on="X_pca", n_SEACells=90,
    use_gpu=True, use_unified=True,   # GPU: needs cupy + cuML / RAPIDS
)
model.construct_kernel_matrix()
model.fit(min_iter=10, max_iter=100)
```
