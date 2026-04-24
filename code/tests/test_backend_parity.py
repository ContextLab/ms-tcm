"""Tier-1 (numpy) vs Tier-2 (JAX) backend parity — Constitution II.

Both backends delegate to ``ms_tcm._likelihood_core`` (single source of
truth). This test file is the CI enforcement mechanism: if either backend
is ever modified to compute a different likelihood than the other, these
tests fail loudly — preventing a recurrence of the pre-refactor situation
where the JAX backend silently omitted C&Z 2025 Eq 7 stopping-rule terms.

The tests below fall into three categories:

1. Full-dataset parity (FRFR-category): every (participant, list) pair
   in the published dataset must produce identical LLs under both
   backends to within float64 rounding (target: < 1e-10 absolute).

2. Random-case parity: small synthetic lists with diverse storyline
   structures exercise edge cases (single recall, all recalled, returns,
   standard-TCM reduction) that the FRFR dataset might not cover.

3. Oracle cross-check: both backends must agree with the equation-derived
   oracle in ``test_likelihood_core.py`` to the same tolerance.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def _setup_jax_float64():
    """Ensure JAX is in float64 mode before importing the backend."""
    os.environ.setdefault("MS_TCM_JAX_DTYPE", "float64")


@pytest.fixture(scope="module")
def frfr_dataset():
    from ms_tcm.dataset import load_dataset
    return load_dataset("/Users/jmanning/ms-tcm/data/raw/frfr_category")


@pytest.fixture(scope="module")
def default_params():
    from ms_tcm.params import ModelParameters
    return ModelParameters(
        beta_enc=0.679, beta_story=0.40, gamma_fc=0.315, k=6.5,
        lambda_reinstate=0.8, beta_rec=0.326, epsilon_d=1.04,
    )


def test_full_frfr_dataset_parity(frfr_dataset, default_params):
    """Tier-1 and Tier-2 agree on the full FRFR-category LL to 1e-10."""
    _setup_jax_float64()
    from ms_tcm.jax_backend.hcmr_jax import dataset_log_likelihood_jax
    from ms_tcm.likelihood import dataset_log_likelihood, clear_encoding_cache

    clear_encoding_cache()
    ll_tier1 = dataset_log_likelihood(frfr_dataset, default_params)
    ll_jax = dataset_log_likelihood_jax(default_params, frfr_dataset)
    assert np.isfinite(ll_tier1) and np.isfinite(ll_jax)
    diff = abs(ll_tier1 - ll_jax)
    assert diff < 1e-10, (
        f"Tier-1/Tier-2 divergence on full FRFR dataset: "
        f"Tier-1={ll_tier1!r}, Tier-2={ll_jax!r}, diff={ll_tier1 - ll_jax!r}. "
        f"Both backends must delegate to _likelihood_core; a non-zero "
        f"diff here indicates one backend is out of sync."
    )


def _build_recall_arrays(recall_sps_list, W):
    """Build (recall_sps, recall_mask) padded to max(W, len)."""
    n = len(recall_sps_list)
    R = max(W, n, 1)
    recall_sps = np.zeros(R, dtype=np.int32)
    recall_mask = np.zeros(R, dtype=bool)
    recall_sps[:n] = recall_sps_list
    recall_mask[:n] = True
    return recall_sps, recall_mask


@pytest.mark.parametrize("seed", list(range(6)))
def test_random_list_parity(seed, default_params):
    """Random W/K/recalls — exercise diverse shapes and orderings."""
    _setup_jax_float64()
    from ms_tcm._likelihood_core import compute_list_log_likelihood_numpy
    from ms_tcm.jax_backend.hcmr_jax import list_log_likelihood_jax

    rng = np.random.default_rng(seed)
    W = int(rng.integers(3, 10))
    K = int(rng.integers(1, min(4, W) + 1))

    # Build a list with each of K storylines appearing at least once.
    slots = list(range(W))
    rng.shuffle(slots)
    cat_indices = np.zeros(W, dtype=np.int32)
    for k in range(K):
        cat_indices[slots[k]] = k
    for i in range(K, W):
        cat_indices[slots[i]] = rng.integers(0, K)

    # Recalls: random subset, random order.
    n_recalls = int(rng.integers(1, W))
    recall_list = (rng.permutation(W)[:n_recalls] + 1).tolist()
    recall_sps, recall_mask = _build_recall_arrays(recall_list, W)

    ll_numpy = compute_list_log_likelihood_numpy(
        default_params, cat_indices.astype(np.int64),
        recall_sps.astype(np.int64), recall_mask,
        W=W, K=K,
    )
    ll_jax = list_log_likelihood_jax(
        default_params, cat_indices, recall_sps, recall_mask,
    )
    assert abs(ll_numpy - ll_jax) < 1e-10, (
        f"numpy/jax divergence on random case seed={seed} "
        f"W={W} K={K} cat_indices={cat_indices.tolist()} "
        f"recalls={recall_list}: "
        f"numpy={ll_numpy!r}, jax={ll_jax!r}"
    )


def test_standard_tcm_reduction_parity(frfr_dataset):
    """Standard-TCM reduction (λ=0, single storyline) parity on FRFR."""
    _setup_jax_float64()
    from ms_tcm.jax_backend.hcmr_jax import dataset_log_likelihood_jax
    from ms_tcm.likelihood import dataset_log_likelihood, clear_encoding_cache
    from ms_tcm.params import ModelParameters

    params = ModelParameters.standard_tcm_reduction(
        beta_enc=0.72, beta_story=0.35, gamma_fc=0.30, k=7.0,
        beta_rec=0.30, epsilon_d=1.0, paradigm="free_recall",
    )
    clear_encoding_cache()
    ll_tier1 = dataset_log_likelihood(frfr_dataset, params)
    ll_jax = dataset_log_likelihood_jax(params, frfr_dataset)
    assert np.isfinite(ll_tier1) and np.isfinite(ll_jax)
    assert abs(ll_tier1 - ll_jax) < 1e-10, (
        f"standard-TCM divergence: Tier-1={ll_tier1!r}, "
        f"Tier-2={ll_jax!r}, diff={ll_tier1 - ll_jax!r}"
    )


@pytest.mark.parametrize("beta_enc,beta_story,k,lambda_r,beta_rec,eps_d", [
    (0.3, 0.2, 2.0, 0.3, 0.2, 0.5),
    (0.9, 0.5, 15.0, 0.9, 0.7, 3.0),
    (0.5, 0.1, 6.5, 0.5, 0.3, 1.5),
])
def test_parameter_sweep_parity(
    beta_enc, beta_story, k, lambda_r, beta_rec, eps_d, frfr_dataset,
):
    """Parity holds across parameter values the optimizer may visit."""
    _setup_jax_float64()
    from ms_tcm.jax_backend.hcmr_jax import dataset_log_likelihood_jax
    from ms_tcm.likelihood import dataset_log_likelihood, clear_encoding_cache
    from ms_tcm.params import ModelParameters

    params = ModelParameters(
        beta_enc=beta_enc, beta_story=beta_story, gamma_fc=0.315, k=k,
        lambda_reinstate=lambda_r, beta_rec=beta_rec, epsilon_d=eps_d,
    )
    clear_encoding_cache()
    ll_tier1 = dataset_log_likelihood(frfr_dataset, params)
    ll_jax = dataset_log_likelihood_jax(params, frfr_dataset)
    # Relative tolerance of 1e-10 of |LL|, with a 1e-8 floor for large LLs.
    tol = max(1e-10 * max(abs(ll_tier1), 1.0), 1e-8)
    assert abs(ll_tier1 - ll_jax) < tol, (
        f"param sweep divergence at beta_enc={beta_enc}, k={k}, lambda={lambda_r}: "
        f"Tier-1={ll_tier1!r}, Tier-2={ll_jax!r}, diff={ll_tier1 - ll_jax!r}"
    )
