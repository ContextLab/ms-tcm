"""Tier-1 (numpy) vs Tier-2 (JAX) backend parity for ``_likelihood_core``.

Both backends call the same ``compute_list_log_likelihood`` function with
different ``xp`` namespaces (``numpy`` vs ``jax.numpy``). This test file
asserts that under float64 they produce bit-identical log-likelihoods on
random inputs — Constitution II (Single Source of Truth) enforcement.

Pre-refactor history: prior versions of the JAX backend silently omitted
C&Z 2025 stopping-rule terms, drifting from Tier-1 by ~1 log-unit per
recall on a 3-pt subset. The xp-dispatch refactor + parity tests prevent
recurrence.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def _setup_jax_float64():
    os.environ.setdefault("MS_TCM_JAX_DTYPE", "float64")


@pytest.fixture(scope="module")
def default_params():
    from ms_tcm.params import ModelParameters
    return ModelParameters(
        beta_enc=0.679, beta_story=0.400, gamma_fc=0.315, k=6.50,
        beta_rec=0.326, epsilon_d=1.04, beta_rein=0.300,
        lambda_reinstate=0.0, paradigm="free_recall",
    )


def _build_recall_arrays(rng, W, R=None):
    if R is None:
        R = int(rng.integers(1, W + 1))
    sps = (rng.permutation(W)[:R] + 1).tolist()
    R_pad = max(R, W)
    recall_sps = np.zeros(R_pad, dtype=np.int64)
    recall_mask = np.zeros(R_pad, dtype=bool)
    recall_sps[:R] = sps
    recall_mask[:R] = True
    return recall_sps, recall_mask, R


@pytest.mark.parametrize("seed", list(range(8)))
def test_jax_numpy_parity_random(seed, default_params):
    """numpy and JAX (float64) compute identical LLs on random small lists."""
    _setup_jax_float64()
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from ms_tcm._likelihood_core import (
        CoreHyperparams,
        compute_list_log_likelihood,
        compute_list_log_likelihood_numpy,
    )

    rng = np.random.default_rng(seed)
    W = int(rng.integers(4, 12))
    recall_sps, recall_mask, R = _build_recall_arrays(rng, W)

    ll_numpy = compute_list_log_likelihood_numpy(
        default_params, recall_sps, recall_mask, W=W,
    )
    hp = CoreHyperparams.from_model_parameters(default_params)
    ll_jax = float(compute_list_log_likelihood(
        hp, W=W,
        recall_sps=jnp.asarray(recall_sps, dtype=jnp.int32),
        recall_mask=jnp.asarray(recall_mask, dtype=jnp.bool_),
        xp=jnp, dtype=jnp.float64,
    ))
    assert abs(ll_numpy - ll_jax) < 1e-10, (
        f"seed={seed} W={W} R={R}: numpy={ll_numpy!r}, jax={ll_jax!r}, "
        f"diff={ll_numpy - ll_jax!r}"
    )


@pytest.mark.parametrize("beta_enc,beta_list,k,beta_rec,eps_d,gamma_fc", [
    (0.3, 0.2, 2.0, 0.2, 0.5, 0.1),
    (0.9, 0.5, 15.0, 0.7, 3.0, 0.9),
    (0.5, 0.2, 6.5, 0.3, 1.5, 0.5),
    (0.679, 0.400, 6.50, 0.326, 1.04, 0.315),  # C&Z Table 1
])
def test_jax_numpy_parity_param_sweep(
    beta_enc, beta_list, k, beta_rec, eps_d, gamma_fc,
):
    """Parity holds across the parameter range an optimizer might visit."""
    _setup_jax_float64()
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from ms_tcm._likelihood_core import (
        CoreHyperparams,
        compute_list_log_likelihood,
        compute_list_log_likelihood_numpy,
    )
    from ms_tcm.params import ModelParameters

    params = ModelParameters(
        beta_enc=beta_enc, beta_story=beta_list, gamma_fc=gamma_fc, k=k,
        beta_rec=beta_rec, epsilon_d=eps_d, beta_rein=0.300,
        lambda_reinstate=0.0, paradigm="free_recall",
    )
    rng = np.random.default_rng(7)
    W = 6
    recall_sps, recall_mask, R = _build_recall_arrays(rng, W, R=4)

    ll_numpy = compute_list_log_likelihood_numpy(
        params, recall_sps, recall_mask, W=W,
    )
    hp = CoreHyperparams.from_model_parameters(params)
    ll_jax = float(compute_list_log_likelihood(
        hp, W=W,
        recall_sps=jnp.asarray(recall_sps, dtype=jnp.int32),
        recall_mask=jnp.asarray(recall_mask, dtype=jnp.bool_),
        xp=jnp, dtype=jnp.float64,
    ))
    tol = max(1e-10 * max(abs(ll_numpy), 1.0), 1e-10)
    assert abs(ll_numpy - ll_jax) < tol, (
        f"params=(β_enc={beta_enc}, β_list={beta_list}, k={k}, β_rec={beta_rec}, "
        f"ε_d={eps_d}, γ_fc={gamma_fc}): numpy={ll_numpy!r}, "
        f"jax={ll_jax!r}, diff={ll_numpy - ll_jax!r}"
    )
