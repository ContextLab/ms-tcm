"""MLE fitting for v6 hierarchical CMR.

Reparameterization (contracts/fitter.md §3):
- beta_enc ∈ (0,1) → logit
- beta_story/beta_enc ∈ (0,1) → logit (ensures beta_enc > beta_story)
- gamma_fc ∈ [0,1] → logit with eps clamp
- k > 0 → log
- lambda_reinstate ∈ [0,1] → logit with eps clamp
- beta_rec ∈ (0,1) → logit (free-recall only)
- epsilon_d > 0 → log (free-recall only)

L-BFGS-B with finite-difference jacobian and ``n_restarts`` random starts
drawn around the C&Z 2025 Table 1 defaults.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.optimize import minimize
from scipy.special import expit, logit

from ms_tcm import __version__
from ms_tcm.dataset import Dataset
from ms_tcm.likelihood import (
    clear_encoding_cache,
    dataset_log_likelihood,
    likelihood_diagnostics,
)
from ms_tcm.params import ModelParameters


class FitError(RuntimeError):
    """Raised when a fit cannot complete (non-convergence, invalid input)."""


_EPS = 1e-6
_FREE_RECALL_PARAMS = (
    "beta_enc", "beta_story_ratio", "gamma_fc", "k", "lambda_reinstate",
    "beta_rec", "epsilon_d",
)
_CUED_RECALL_PARAMS = (
    "beta_enc", "beta_story_ratio", "gamma_fc", "k", "lambda_reinstate",
)


def _theta_from_params(p: ModelParameters) -> np.ndarray:
    """Map ModelParameters to the unconstrained theta vector."""
    # beta_story_ratio = beta_story / beta_enc, in (0, 1) by construction.
    ratio = p.beta_story / p.beta_enc
    theta = [
        logit(p.beta_enc),
        logit(max(_EPS, min(1.0 - _EPS, ratio))),
        logit(max(_EPS, min(1.0 - _EPS, p.gamma_fc))),
        np.log(p.k),
        logit(max(_EPS, min(1.0 - _EPS, p.lambda_reinstate))),
    ]
    if p.paradigm == "free_recall":
        theta.extend([logit(p.beta_rec), np.log(p.epsilon_d)])
    return np.asarray(theta, dtype=np.float64)


def _params_from_theta(
    theta: np.ndarray, *, standard_tcm: bool, paradigm: str, seed: int,
    feature_dim: int,
) -> ModelParameters:
    """Map unconstrained theta back to a validated ModelParameters.

    Clamps ``theta`` scalars before applying the log / logit inverse to keep
    ``np.exp`` out of the overflow regime and to keep ``expit`` numerically
    meaningful. An unclamped optimizer wandering into theta ~ ±40 otherwise
    produces ``k = inf`` and downstream NaN activations (scipy.special
    ``_logsumexp`` then emits ``invalid value encountered in subtract``).
    """
    # Clamp the theta scalars that feed np.exp (k, epsilon_d) to a safe range;
    # logit arguments are clamped separately via ``_EPS`` below. 30 is far
    # outside any psychologically plausible regime for k and epsilon_d and
    # prevents the ``exp`` overflow RuntimeWarning.
    _THETA_CLAMP = 30.0
    beta_enc = float(expit(theta[0]))
    beta_story_ratio = float(expit(theta[1]))
    beta_story = beta_enc * beta_story_ratio
    # Clamp to avoid hitting the beta_enc > beta_story boundary at exactly equality.
    beta_story = min(beta_story, beta_enc - 1e-10)
    gamma_fc = float(expit(theta[2]))
    k = float(np.exp(float(np.clip(theta[3], -_THETA_CLAMP, _THETA_CLAMP))))
    lambda_reinstate = 0.0 if standard_tcm else float(expit(theta[4]))

    if paradigm == "free_recall":
        beta_rec = float(expit(theta[5]))
        epsilon_d = float(np.exp(float(np.clip(theta[6], -_THETA_CLAMP, _THETA_CLAMP))))
    else:
        beta_rec = 0.326  # unused; initialize at C&Z default for validation pass-through
        epsilon_d = 1.04

    # Clamp each parameter into the valid interior so __post_init__ doesn't raise.
    beta_enc = float(max(_EPS, min(1.0 - _EPS, beta_enc)))
    beta_story = float(max(_EPS, min(beta_enc - 1e-10, beta_story)))
    gamma_fc = float(max(0.0, min(1.0, gamma_fc)))
    lambda_reinstate = float(max(0.0, min(1.0, lambda_reinstate)))
    if paradigm == "free_recall":
        beta_rec = float(max(_EPS, min(1.0 - _EPS, beta_rec)))
        epsilon_d = float(max(_EPS, epsilon_d))

    return ModelParameters(
        beta_enc=beta_enc, beta_story=beta_story, gamma_fc=gamma_fc, k=k,
        lambda_reinstate=lambda_reinstate, beta_rec=beta_rec, epsilon_d=epsilon_d,
        feature_dim=feature_dim, paradigm=paradigm, standard_tcm=standard_tcm,
        seed=seed,
    )


@dataclass(frozen=True)
class FitResult:
    """Output of a fit: per-parameter MLE + optional bootstrap CI + metadata."""
    parameters: dict[str, dict[str, float | str | int]]
    log_likelihood: float
    aic: float
    bic: float
    n_participants: int
    n_lists: int
    n_recalls_used: int
    n_intrusions_excluded: int
    seed: int
    elapsed_seconds: float
    ms_tcm_version: str
    dataset_manifest_sha256: str
    standard_tcm: bool
    backend: str = "tier1"
    bootstrap_draws: pa.Table | None = None


def _negative_log_likelihood(
    theta: np.ndarray,
    dataset: Dataset,
    *,
    standard_tcm: bool,
    paradigm: str,
    feature_dim: int,
    seed: int,
) -> float:
    try:
        params = _params_from_theta(
            theta, standard_tcm=standard_tcm, paradigm=paradigm, seed=seed,
            feature_dim=feature_dim,
        )
    except ValueError:
        return 1e12  # invalid parameter region; keep optimizer away
    ll = dataset_log_likelihood(dataset, params)
    if not np.isfinite(ll):
        return 1e12
    return -ll


def fit_mle(
    dataset: Dataset,
    *,
    n_restarts: int = 5,
    seed: int = 0,
    standard_tcm: bool = False,
    paradigm: str = "free_recall",
) -> FitResult:
    """L-BFGS-B MLE with random restarts (Tier 1 path).

    Clears the module-level encoding cache on entry so that two back-to-back
    ``fit_mle(ds, seed=S)`` calls with identical inputs are guaranteed to
    produce bit-identical outputs (FR-023 / test_tier1_mle_within_1e_minus_10).
    Without this, cache population from a prior fit could alter eviction
    ordering during the next fit's optimizer sweep.
    """
    clear_encoding_cache()
    t0 = time.perf_counter()
    feature_dim = dataset.feature_dim

    # Starting theta at C&Z Table 1 defaults.
    default_params = ModelParameters(paradigm=paradigm, feature_dim=feature_dim)
    theta_init = _theta_from_params(default_params)
    n_theta = len(theta_init)
    # Under --standard-tcm the lambda_reinstate coordinate is pinned to 0 via
    # ``_params_from_theta`` (not by changing n_theta here). The optimizer
    # will still vary theta[4], but _params_from_theta overrides it with 0.0
    # before constructing ModelParameters, so the objective is flat in that
    # direction and L-BFGS-B converges cleanly. This matches the FR-011
    # reduction semantics and is verified by test_hcmr_standard_tcm.

    rng = np.random.default_rng(seed)
    best_nll = np.inf
    best_theta = theta_init.copy()

    for r in range(max(1, int(n_restarts))):
        if r == 0:
            theta0 = theta_init.copy()
        else:
            theta0 = theta_init + rng.normal(0, 0.3, size=n_theta)
        res = minimize(
            _negative_log_likelihood, theta0,
            args=(dataset,),
            kwargs={
                "standard_tcm": standard_tcm,
                "paradigm": paradigm,
                "feature_dim": feature_dim,
                "seed": seed,
            } if False else (),
            # scipy.optimize.minimize doesn't accept kwargs directly; use a closure instead.
            method="L-BFGS-B",
        ) if False else _minimize_wrapper(
            dataset, theta0, standard_tcm, paradigm, feature_dim, seed,
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
    free_param_names = _CUED_RECALL_PARAMS if paradigm == "cued_recall" else _FREE_RECALL_PARAMS
    if standard_tcm:
        free_param_names = tuple(n for n in free_param_names if n != "lambda_reinstate")
    k_free = len(free_param_names)
    n = max(1, diag.n_recalls_used)
    aic = 2 * k_free - 2 * ll
    bic = k_free * np.log(n) - 2 * ll

    # Package per-parameter point estimates (no CI from this minimal fit_mle).
    param_dict: dict[str, dict[str, float | str | int]] = {}
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

    import hashlib
    manifest_bytes = str(dataset.manifest).encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

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
        backend="tier1",
    )


def _minimize_wrapper(dataset, theta0, standard_tcm, paradigm, feature_dim, seed):
    """Wrap scipy.optimize.minimize to inject kwargs via a closure."""
    def f(theta):
        return _negative_log_likelihood(
            theta, dataset, standard_tcm=standard_tcm, paradigm=paradigm,
            feature_dim=feature_dim, seed=seed,
        )
    return minimize(f, theta0, method="L-BFGS-B")
