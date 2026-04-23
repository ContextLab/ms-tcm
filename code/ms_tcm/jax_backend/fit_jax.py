"""JAX-native MLE fit driver for Tier 2 (FR-032 / research §R3).

Uses ``jax.grad(negative_log_likelihood)`` for analytic gradients (replaces
scipy L-BFGS-B's finite-difference jacobian) and drives the optimizer via
``scipy.optimize.minimize(..., jac=grad_fn, method='L-BFGS-B')``.

We intentionally do NOT use Optax's L-BFGS because scipy's L-BFGS-B has
battle-hardened line-search + box-constrained behavior that matches the
Tier-1 optimizer exactly; the value-add of the JAX backend is the
gradient (7× savings per optimizer step on a 7-dim parameter vector)
plus JIT-compiled inner-loop evaluation, not the optimizer itself.

Dtype: inherited from ``hcmr_jax._DTYPE`` (float64 default; float32 when
``MS_TCM_JAX_DTYPE=float32``).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from ms_tcm import __version__
from ms_tcm.dataset import Dataset
from ms_tcm.fit import FitResult, _params_from_theta, _theta_from_params
from ms_tcm.likelihood import clear_encoding_cache, likelihood_diagnostics
from ms_tcm.params import ModelParameters


def _build_jax_dataset_negloglik(
    dataset: Dataset, *, standard_tcm: bool, paradigm: str, feature_dim: int,
    seed: int,
):
    """Return a (nll_fn, grad_fn) pair that evaluate the negative log-likelihood
    and its gradient w.r.t. theta, using the JAX backend.

    The closure materializes the per-list inputs (cat_indices, recall_sps,
    recall_mask) once so each nll call just calls the jit'd per-list
    likelihood N_lists times — no dataset-parsing overhead inside the
    optimizer loop.
    """
    import jax
    import jax.numpy as jnp
    from ms_tcm.jax_backend.hcmr_jax import (
        _DTYPE, _make_list_log_likelihood_fn,
    )

    pdf = dataset.presented.to_pandas()
    rdf = dataset.recalled.to_pandas()
    keys = sorted(set(zip(pdf["participant"].tolist(), pdf["list"].tolist())))

    # Determine W, K for each list (FRFR-category: W=16, K=4 uniformly).
    # Group to fixed-shape batches.
    max_recall_len = int(
        rdf[rdf["serial_position"] > 0]
        .groupby(["participant", "list"])
        .size().max()
        if not rdf.empty else 1
    )

    list_records: list[tuple] = []  # (W, K, cat_indices, recall_sps, recall_mask)
    for part, lst in keys:
        pres_sub = pdf[(pdf["participant"] == part) & (pdf["list"] == lst)].sort_values("serial_position")
        rec_sub = rdf[(rdf["participant"] == part) & (rdf["list"] == lst)].sort_values("output_position")
        cats = list(dict.fromkeys(pres_sub["category"].tolist()))
        cat_to_idx = {c: i for i, c in enumerate(cats)}
        cat_indices = jnp.asarray(
            [cat_to_idx[c] for c in pres_sub["category"].tolist()],
            dtype=jnp.int32,
        )
        W = int(cat_indices.shape[0])
        K = int(len(cats))
        rec_rows = rec_sub[rec_sub["serial_position"] > 0]
        sps = rec_rows["serial_position"].to_numpy(dtype=np.int32)
        R = max_recall_len
        recall_sps = np.zeros(R, dtype=np.int32)
        recall_mask = np.zeros(R, dtype=np.bool_)
        n = min(len(sps), R)
        recall_sps[:n] = sps[:n]
        recall_mask[:n] = True
        list_records.append((
            W, K,
            cat_indices,
            jnp.asarray(recall_sps, dtype=jnp.int32),
            jnp.asarray(recall_mask, dtype=jnp.bool_),
        ))

    # Build per-shape JIT'd likelihood functions once.
    shape_fns: dict = {}
    for W, K, _, _, _ in list_records:
        shape_key = (W, K, bool(standard_tcm))
        if shape_key not in shape_fns:
            shape_fns[shape_key] = _make_list_log_likelihood_fn(
                W, K, bool(standard_tcm),
            )

    def nll(theta_np: np.ndarray) -> float:
        theta = jnp.asarray(theta_np, dtype=_DTYPE)
        total = jnp.asarray(0.0, dtype=_DTYPE)
        for W, K, cat_indices, recall_sps, recall_mask in list_records:
            fn = shape_fns[(W, K, bool(standard_tcm))]
            total = total + fn(theta, cat_indices, recall_sps, recall_mask)
        return float(-total)

    # Build a gradient function using jax.grad over a single-list summand,
    # then stack contributions.
    def _sum_ll(theta):
        total = jnp.asarray(0.0, dtype=_DTYPE)
        for W, K, cat_indices, recall_sps, recall_mask in list_records:
            fn = shape_fns[(W, K, bool(standard_tcm))]
            total = total + fn(theta, cat_indices, recall_sps, recall_mask)
        return total

    grad_ll = jax.jit(jax.grad(_sum_ll))

    def grad_nll(theta_np: np.ndarray) -> np.ndarray:
        theta = jnp.asarray(theta_np, dtype=_DTYPE)
        g = grad_ll(theta)
        return -np.asarray(g, dtype=np.float64)

    return nll, grad_nll


def fit_mle_jax(
    dataset: Dataset,
    *,
    n_restarts: int = 5,
    seed: int = 0,
    standard_tcm: bool = False,
    paradigm: str = "free_recall",
) -> FitResult:
    """L-BFGS-B MLE with JAX-native log-likelihood + analytic gradients."""
    from ms_tcm.jax_backend.hcmr_jax import _JAX_DTYPE_ENV

    clear_encoding_cache()
    t0 = time.perf_counter()
    feature_dim = dataset.feature_dim

    default_params = ModelParameters(paradigm=paradigm, feature_dim=feature_dim)
    theta_init = _theta_from_params(default_params)
    n_theta = len(theta_init)

    rng = np.random.default_rng(seed)
    best_nll = np.inf
    best_theta = theta_init.copy()

    nll_fn, grad_fn = _build_jax_dataset_negloglik(
        dataset, standard_tcm=standard_tcm, paradigm=paradigm,
        feature_dim=feature_dim, seed=seed,
    )

    for r in range(max(1, int(n_restarts))):
        if r == 0:
            theta0 = theta_init.copy()
        else:
            theta0 = theta_init + rng.normal(0, 0.3, size=n_theta)
        res = minimize(
            nll_fn, theta0,
            jac=grad_fn,
            method="L-BFGS-B",
        )
        if res.fun < best_nll:
            best_nll = float(res.fun)
            best_theta = res.x.copy()

    mle_params = _params_from_theta(
        best_theta, standard_tcm=standard_tcm, paradigm=paradigm, seed=seed,
        feature_dim=feature_dim,
    )
    ll = -best_nll

    diag = likelihood_diagnostics(dataset)
    k_free = 7 if paradigm == "free_recall" else 5
    if standard_tcm:
        k_free -= 1
    n = max(1, diag.n_recalls_used)
    aic = 2 * k_free - 2 * ll
    bic = k_free * np.log(n) - 2 * ll

    # Package per-parameter point estimates.
    import hashlib
    manifest_hash = hashlib.sha256(
        str(dataset.manifest).encode("utf-8")
    ).hexdigest()
    param_dict: dict = {}
    for name in ("beta_enc", "beta_story", "gamma_fc", "k", "lambda_reinstate",
                 "beta_rec", "epsilon_d"):
        val = getattr(mle_params, name)
        param_dict[name] = {
            "mle": float(val),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "ci_method": "none",
            "n_bootstraps": 0,
            "n_converged": 0,
        }

    backend_name = f"tier2-{_JAX_DTYPE_ENV}"

    return FitResult(
        parameters=param_dict,
        log_likelihood=float(ll),
        aic=float(aic),
        bic=float(bic),
        n_participants=int(
            len(set(dataset.presented.to_pandas()["participant"].tolist()))
        ),
        n_lists=int(diag.n_lists),
        n_recalls_used=int(diag.n_recalls_used),
        n_intrusions_excluded=int(diag.n_intrusions_excluded),
        seed=int(seed),
        elapsed_seconds=float(time.perf_counter() - t0),
        ms_tcm_version=str(__version__),
        dataset_manifest_sha256=manifest_hash,
        standard_tcm=bool(standard_tcm),
        backend=backend_name,
    )
