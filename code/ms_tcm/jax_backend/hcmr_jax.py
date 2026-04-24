"""JAX-native hierarchical CMR encoder + likelihood (Tier-2 / FR-032).

This module is a thin wrapper around ``ms_tcm._likelihood_core`` driven
with ``xp=jax.numpy``. The math is identical to Tier-1 (same core module,
same functions) — this backend differs only in that it:

- runs under ``jax.numpy`` for autodiff + accelerator support,
- compiles encoding + retrieval into ``jax.lax.scan``-based kernels,
- ``jax.jit``-caches the compiled function per ``(W, K, standard_tcm)``.

Because both Tier-1 and Tier-2 delegate to the same ``_likelihood_core``,
they produce bit-identical log-likelihoods for the same inputs under
float64. This is enforced by ``tests/test_backend_parity.py`` and by the
equation-derived oracle tests in ``tests/test_likelihood_core.py``.

Dtype is controlled by the ``MS_TCM_JAX_DTYPE`` env var at import time:

- ``float64`` (default): flips ``jax_enable_x64`` on; bit-identical to
  Tier-1 numpy float64 within 1e-10 across the full FRFR dataset.
- ``float32``: leaves jax in its default float32 mode; parity with
  Tier-1 within ~1e-5 (sufficient for gradient-based MLE).
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from ms_tcm._likelihood_core import (
    CoreHyperparams,
    compute_list_log_likelihood,
)
from ms_tcm.dataset import Dataset
from ms_tcm.params import ModelParameters


# --- dtype configuration -------------------------------------------------

_JAX_DTYPE_ENV = os.environ.get("MS_TCM_JAX_DTYPE", "float64").lower()
if _JAX_DTYPE_ENV == "float64":
    jax.config.update("jax_enable_x64", True)
    _DTYPE = jnp.float64
elif _JAX_DTYPE_ENV == "float32":
    _DTYPE = jnp.float32
else:
    raise RuntimeError(
        f"MS_TCM_JAX_DTYPE must be 'float64' or 'float32'; got {_JAX_DTYPE_ENV!r}"
    )


def get_jax_dtype():
    """Return the currently-configured JAX dtype."""
    return _DTYPE


# --- theta ↔ params helpers ----------------------------------------------
#
# ``fit_jax.py`` drives the optimizer over an unconstrained ``theta`` vector
# that maps into box-constrained ``ModelParameters`` via the same
# reparameterization used in Tier-1 (``ms_tcm.fit._theta_from_params``).
# We replicate it in JAX to keep gradients through the mapping.


def _theta_to_hyperparams(theta, standard_tcm: bool):
    """Map a 7-vector ``theta`` to the fields of ``CoreHyperparams``.

    Matches ``ms_tcm.fit._params_from_theta``: theta[0]=logit(β_enc),
    theta[1]=logit(β_story/β_enc), theta[2]=logit(γ_fc), theta[3]=log(k),
    theta[4]=logit(λ), theta[5]=logit(β_rec), theta[6]=log(ε_d).
    """
    from jax.scipy.special import expit as jexpit

    beta_enc = jexpit(theta[0])
    ratio = jexpit(theta[1])
    beta_story = beta_enc * ratio
    beta_story = jnp.minimum(beta_story, beta_enc - 1e-10)
    gamma_fc = jexpit(theta[2])
    k = jnp.exp(jnp.clip(theta[3], -30.0, 30.0))
    lambda_reinstate = jnp.where(standard_tcm, 0.0, jexpit(theta[4]))
    beta_rec = jexpit(theta[5])
    epsilon_d = jnp.exp(jnp.clip(theta[6], -30.0, 30.0))
    return beta_enc, beta_story, gamma_fc, k, lambda_reinstate, beta_rec, epsilon_d


# --- JIT-compiled per-list likelihood ------------------------------------


def _make_list_log_likelihood_fn(W: int, K: int, standard_tcm: bool):
    """Return a ``jax.jit``-compiled per-list log-likelihood function.

    Signature: ``fn(theta, cat_indices, recall_sps, recall_mask) -> float``.

    Both encoding and retrieval are driven by ``_likelihood_core`` with
    ``xp=jax.numpy`` — so the math is guaranteed identical to Tier-1.
    The scan-friendly step functions in the core are the single source
    of truth for all model equations.
    """
    phi = 30.0 if standard_tcm else 1.5
    psi = 0.8 if standard_tcm else 0.5

    def list_log_likelihood(theta, cat_indices, recall_sps, recall_mask):
        theta = theta.astype(_DTYPE)
        beta_enc, beta_story, gamma_fc, k, lambda_reinstate, beta_rec, epsilon_d = (
            _theta_to_hyperparams(theta, standard_tcm)
        )
        hp = CoreHyperparams(
            beta_enc=beta_enc,
            beta_story=beta_story,
            gamma_fc=gamma_fc,
            k=k,
            lambda_reinstate=lambda_reinstate,
            beta_rec=beta_rec,
            epsilon_d=epsilon_d,
            standard_tcm=bool(standard_tcm),
            phi=phi,
            psi=psi,
        )
        return compute_list_log_likelihood(
            hp, cat_indices, recall_sps, recall_mask,
            W=W, K=K, xp=jnp, dtype=_DTYPE,
        )

    return jax.jit(list_log_likelihood)


def _make_encode_fn(W: int, K: int, standard_tcm: bool):
    """Return a ``jit``-compiled encode function (encodes + returns c_item_traj, m_ic).

    Kept for backwards compat with ``test_jax_backend.py::test_jax_encode_*``.
    Under the refactor, the core's ``run_encoding`` is the single source
    of truth; this wrapper exposes it under the pre-refactor signature.
    """
    phi = 30.0 if standard_tcm else 1.5
    psi = 0.8 if standard_tcm else 0.5
    from ms_tcm._likelihood_core import run_encoding

    def encode(beta_enc, beta_story, lambda_reinstate, cat_indices):
        hp = CoreHyperparams(
            beta_enc=beta_enc.astype(_DTYPE),
            beta_story=beta_story.astype(_DTYPE),
            gamma_fc=0.315,  # unused under identity M^FC_pre
            k=6.5,           # unused in encoding
            lambda_reinstate=lambda_reinstate.astype(_DTYPE),
            beta_rec=0.326,  # unused in encoding
            epsilon_d=1.04,  # unused in encoding
            standard_tcm=bool(standard_tcm),
            phi=phi,
            psi=psi,
        )
        return run_encoding(hp, cat_indices, W, K, jnp, _DTYPE)

    return jax.jit(encode)


# --- module-level caches -------------------------------------------------


_ENCODE_FN_CACHE: dict = {}
_LIST_LL_FN_CACHE: dict = {}


def encode_list_jax(
    params: ModelParameters, cat_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper that evaluates the JIT'd encode for a single list."""
    W = cat_indices.shape[0]
    K = int(cat_indices.max()) + 1 if W > 0 else 1
    key = (W, K, bool(params.standard_tcm))
    fn = _ENCODE_FN_CACHE.get(key)
    if fn is None:
        fn = _make_encode_fn(W, K, bool(params.standard_tcm))
        _ENCODE_FN_CACHE[key] = fn
    c_item_traj, m_ic = fn(
        jnp.array(params.beta_enc, dtype=_DTYPE),
        jnp.array(params.beta_story, dtype=_DTYPE),
        jnp.array(params.lambda_reinstate, dtype=_DTYPE),
        jnp.asarray(cat_indices, dtype=jnp.int32),
    )
    return np.asarray(c_item_traj), np.asarray(m_ic)


def list_log_likelihood_jax(
    params: ModelParameters,
    cat_indices: np.ndarray,
    recall_sps: np.ndarray,
    recall_mask: np.ndarray,
) -> float:
    """Evaluate per-list log-likelihood under JAX (single-list entry point)."""
    W = cat_indices.shape[0]
    K = int(cat_indices.max()) + 1 if W > 0 else 1
    key = (W, K, bool(params.standard_tcm))
    fn = _LIST_LL_FN_CACHE.get(key)
    if fn is None:
        fn = _make_list_log_likelihood_fn(W, K, bool(params.standard_tcm))
        _LIST_LL_FN_CACHE[key] = fn
    # Build theta from params (same reparameterization as ms_tcm.fit).
    from scipy.special import logit as slogit

    theta = np.array([
        slogit(np.clip(params.beta_enc, 1e-6, 1.0 - 1e-6)),
        slogit(np.clip(params.beta_story / params.beta_enc, 1e-6, 1.0 - 1e-6)),
        slogit(np.clip(params.gamma_fc, 1e-6, 1.0 - 1e-6)),
        np.log(params.k),
        slogit(np.clip(max(params.lambda_reinstate, 1e-6), 1e-6, 1.0 - 1e-6)),
        slogit(np.clip(params.beta_rec, 1e-6, 1.0 - 1e-6)),
        np.log(params.epsilon_d),
    ], dtype=np.float64)
    result = fn(
        jnp.asarray(theta, dtype=_DTYPE),
        jnp.asarray(cat_indices, dtype=jnp.int32),
        jnp.asarray(recall_sps, dtype=jnp.int32),
        jnp.asarray(recall_mask, dtype=jnp.bool_),
    )
    return float(result)


def dataset_log_likelihood_jax(
    params: ModelParameters, dataset: Dataset,
) -> float:
    """Sum of per-list log-likelihoods via the JAX backend.

    Iterates over (participant, list_) in Python (per-list W varies);
    each list's likelihood is one JIT-compiled call.
    """
    pdf = dataset.presented.to_pandas()
    rdf = dataset.recalled.to_pandas()
    keys = sorted(set(zip(pdf["participant"].tolist(), pdf["list"].tolist())))

    total = 0.0
    # Pad recalls to the global max recall length (across the dataset) so
    # participants who produced more recalls than list length aren't
    # silently truncated. Keep consistent with likelihood._build_recall_arrays.
    if not rdf.empty:
        max_recall_len = int(
            rdf.groupby(["participant", "list"]).size().max()
        )
    else:
        max_recall_len = 1

    for part, lst in keys:
        pres_sub = pdf[(pdf["participant"] == part) & (pdf["list"] == lst)].sort_values("serial_position")
        rec_sub = rdf[(rdf["participant"] == part) & (rdf["list"] == lst)].sort_values("output_position")

        cats = list(dict.fromkeys(pres_sub["category"].tolist()))
        cat_to_idx = {c: i for i, c in enumerate(cats)}
        cat_indices = np.array(
            [cat_to_idx[c] for c in pres_sub["category"].tolist()],
            dtype=np.int32,
        )
        W = cat_indices.shape[0]
        # Drop extra-list intrusions (sp == 0) from the LL — matches Tier-1.
        rec_rows = rec_sub[rec_sub["serial_position"] > 0]
        sps = rec_rows["serial_position"].to_numpy(dtype=np.int32)
        R = max(W, max_recall_len, 1)
        recall_sps = np.zeros(R, dtype=np.int32)
        recall_mask = np.zeros(R, dtype=np.bool_)
        n = len(sps)
        recall_sps[:n] = sps
        recall_mask[:n] = True

        try:
            ll = list_log_likelihood_jax(params, cat_indices, recall_sps, recall_mask)
        except Exception:
            return float("-inf")
        if not np.isfinite(ll):
            return float("-inf")
        total += ll
    return total
