"""Parity + correctness tests for the unified ``SEACells.model.SEACellsModel``.

These encode the guarantees established during the GPU rewrite:

* the unified CPU backend reproduces the legacy ``cpu_dense`` optimizer exactly given the
  same kernel and initialization;
* the memory-scalable reduced-form RSS equals the direct ``||M - MBA||_F``;
* the GPU backend matches the CPU backend, per stage, when both use the same kernel + init
  (GPU tests are skipped automatically when no CUDA device / cupy is available).

Note on the full pipeline: when the CPU and GPU backends each build their *own* kNN kernel,
sklearn and cuML differ at ~1e-6 (dtype-independent), and because kernel archetypal analysis
is non-convex this can amplify to a different-but-equally-valid optimum. Exact CPU/GPU
agreement is therefore only asserted for the *shared-kernel + shared-init* case, which
isolates the optimizer.
"""

import os

import numpy as np
import pytest
import scanpy as sc

from SEACells import core
from SEACells.model import SEACellsModel

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "SEACells",
    "data",
    "sample_data.h5ad",
)

try:
    import cupy as cp  # noqa: F401

    _HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # noqa: BLE001
    _HAS_GPU = False

gpu_required = pytest.mark.skipif(not _HAS_GPU, reason="no CUDA GPU / cupy available")

K_SEACELLS = 10
N_ITERS = 20


@pytest.fixture(scope="module")
def ad():
    return sc.read(DATA)


@pytest.fixture(scope="module")
def fixed_init(ad):
    """A deterministic kernel + archetypes + assignment shared across backends."""
    m = SEACellsModel(ad.copy(), "X_pca", K_SEACELLS, use_gpu=False, verbose=False)
    m.construct_kernel_matrix()
    M = m.kernel_matrix
    n = M.shape[0]
    rng = np.random.RandomState(42)
    arch = rng.choice(n, K_SEACELLS, replace=False)
    A0 = rng.random((K_SEACELLS, n))
    A0 /= A0.sum(0)
    return M, arch, A0


def _fit(use_gpu, M, arch, A0, ad, **kw):
    m = SEACellsModel(
        ad.copy(), "X_pca", K_SEACELLS, use_gpu=use_gpu, verbose=False,
        convergence_epsilon=1e-5, **kw,
    )
    m.add_precomputed_kernel_matrix(M)
    m.fit(min_iter=N_ITERS, max_iter=N_ITERS, initial_archetypes=arch, initial_assignments=A0)
    return m


def test_reduced_rss_matches_frobenius(fixed_init, ad):
    """Reduced-form RSS equals the direct ||M - MBA||_F to machine precision."""
    M, arch, A0 = fixed_init
    m = _fit(False, M, arch, A0, ad)
    A, B = m.A_, m.B_
    R = (M.dot(B)).dot(A)
    direct = np.linalg.norm((M - R))
    assert abs(direct - m.compute_RSS(A, B)) < 1e-8 * direct


def test_unified_cpu_matches_legacy(fixed_init, ad):
    """Unified CPU backend reproduces legacy cpu_dense given identical kernel + init."""
    from SEACells import cpu_dense

    M, arch, A0 = fixed_init
    ref = cpu_dense.SEACellsCPUDense(
        ad.copy(), "X_pca", K_SEACELLS, verbose=False, convergence_epsilon=1e-5
    )
    ref.add_precomputed_kernel_matrix(M)
    ref.fit(min_iter=N_ITERS, max_iter=N_ITERS, initial_archetypes=arch, initial_assignments=A0)

    new = _fit(False, M, arch, A0, ad)
    assert np.abs(np.asarray(new.A_) - np.asarray(ref.A_)).max() < 1e-10
    assert np.abs(np.asarray(new.B_) - np.asarray(ref.B_)).max() < 1e-10
    assert np.abs(np.array(new.RSS_iters) - np.array(ref.RSS_iters)).max() < 1e-9


@gpu_required
def test_gpu_matches_cpu_shared_kernel(fixed_init, ad):
    """GPU optimizer matches CPU given the same kernel + init: identical hard labels."""
    M, arch, A0 = fixed_init
    mc = _fit(False, M, arch, A0, ad)
    mg = _fit(True, M, arch, A0, ad)
    lc = mc.get_hard_assignments()["SEACell"].values
    lg = mg.get_hard_assignments()["SEACell"].values
    assert (lc == lg).mean() == 1.0
    assert abs(mc.RSS_iters[-1] - mg.RSS_iters[-1]) < 1e-3


@gpu_required
def test_gpu_kernel_close_to_cpu(ad):
    """GPU and CPU exact-kNN kernels agree to ~float precision and same sparsity."""
    mc = SEACellsModel(ad.copy(), "X_pca", K_SEACELLS, use_gpu=False, verbose=False)
    mc.construct_kernel_matrix()
    mg = SEACellsModel(ad.copy(), "X_pca", K_SEACELLS, use_gpu=True, verbose=False)
    mg.construct_kernel_matrix()
    Mc = mc.kernel_matrix
    Mg = mg.kernel_matrix.get()
    assert Mc.nnz == Mg.nnz
    assert np.abs((Mc - Mg)).max() < 1e-4


def test_core_factory_routes_to_unified(ad):
    """core.SEACells(use_unified=True) returns the unified model; default stays legacy."""
    m = core.SEACells(ad.copy(), "X_pca", K_SEACELLS, use_unified=True, verbose=False)
    assert isinstance(m, SEACellsModel)
    legacy = core.SEACells(ad.copy(), "X_pca", K_SEACELLS, verbose=False)
    assert type(legacy).__name__ == "SEACellsCPUDense"
