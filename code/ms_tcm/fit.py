"""Maximum-likelihood fitting via L-BFGS-B with random restarts.

See contracts/fitter.md for the contract and data-model.md section 4.4 for the
reparameterization choice (logit / softplus so bounds=None).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize

from ms_tcm import __version__
from ms_tcm.dataset import Dataset
from ms_tcm.io import sha256_file, write_json_sorted
from ms_tcm.likelihood import dataset_log_likelihood, likelihood_diagnostics
from ms_tcm.params import ModelParameters


class FitError(RuntimeError):
    """Raised when the optimizer fails to produce a usable result."""


# --- Reparameterization helpers ------------------------------------------------

def _sigmoid(z: float) -> float:
    # Stable sigmoid.
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    e = np.exp(z)
    return e / (1.0 + e)


def _softplus(z: float) -> float:
    if z > 30:
        return float(z)
    return float(np.log1p(np.exp(z)))


# --- Free-parameter specification ---------------------------------------------

def _free_parameter_names(
    *, standard_tcm: bool, optional_mechanisms: dict[str, bool] | None,
    separate_retrieval_weights: bool,
) -> list[str]:
    if standard_tcm:
        names = ["beta_global"]
    else:
        names = ["beta_global", "beta_storyline", "w_global"]
        if separate_retrieval_weights:
            names.append("w_global_ret")
    # tau (softmax gain) and phi_s / phi_d (primacy gradient) are always
    # free. Without tau the softmax of bounded cosine similarities is close
    # to uniform and the model cannot produce recency or contiguity; without
    # phi_s / phi_d the model has no primacy mechanism at all.
    names.extend(["tau", "phi_s", "phi_d"])
    opt = optional_mechanisms or {}
    if opt.get("gamma"):
        names.append("gamma")
    if opt.get("lambda"):
        names.append("lambda_interference")
    return names


_SIGMOID_PARAMS = ("beta_global", "beta_storyline", "w_global", "w_global_ret")
_SOFTPLUS_PARAMS = ("gamma", "lambda_interference", "tau", "phi_s", "phi_d")


def _pack_theta(
    theta: dict[str, float], free_names: list[str],
) -> np.ndarray:
    """Pack a theta dict into an unconstrained z-vector for the optimizer."""
    z = np.empty(len(free_names), dtype=np.float64)
    for i, name in enumerate(free_names):
        v = theta[name]
        if name in _SIGMOID_PARAMS:
            v = min(max(v, 1e-8), 1 - 1e-8)
            z[i] = float(np.log(v / (1 - v)))
        elif name in _SOFTPLUS_PARAMS:
            v = max(v, 1e-8)
            z[i] = float(np.log(np.expm1(v)))
        else:
            z[i] = float(v)
    return z


def _unpack_theta(
    z: np.ndarray, free_names: list[str], *, standard_tcm: bool,
    separate_retrieval_weights: bool,
) -> dict[str, float]:
    theta: dict[str, float] = {}
    for i, name in enumerate(free_names):
        zi = float(z[i])
        if name in _SIGMOID_PARAMS:
            theta[name] = _sigmoid(zi)
        elif name in _SOFTPLUS_PARAMS:
            theta[name] = _softplus(zi)
        else:
            theta[name] = zi

    # Derived values.
    if standard_tcm:
        theta.setdefault("beta_storyline", 0.5)
        theta["w_global"] = 1.0
        theta["w_storyline"] = 0.0
        theta["w_global_ret"] = 1.0
        theta["w_storyline_ret"] = 0.0
    else:
        theta["w_storyline"] = 1.0 - theta["w_global"]
        if separate_retrieval_weights:
            theta["w_storyline_ret"] = 1.0 - theta["w_global_ret"]
        else:
            theta["w_global_ret"] = theta["w_global"]
            theta["w_storyline_ret"] = theta["w_storyline"]

    theta.setdefault("gamma", 0.0)
    theta.setdefault("lambda_interference", 0.0)
    theta.setdefault("tau", 1.0)
    theta.setdefault("phi_s", 0.0)
    theta.setdefault("phi_d", 1.0)
    return theta


def _theta_to_params(theta: dict[str, float]) -> ModelParameters:
    return ModelParameters(
        beta_global=theta["beta_global"],
        beta_storyline=theta["beta_storyline"],
        w_global=theta["w_global"],
        w_storyline=theta["w_storyline"],
        w_global_ret=theta["w_global_ret"],
        w_storyline_ret=theta["w_storyline_ret"],
        gamma=theta.get("gamma", 0.0),
        lambda_interference=theta.get("lambda_interference", 0.0),
        tau=theta.get("tau", 1.0),
        phi_s=theta.get("phi_s", 0.0),
        phi_d=theta.get("phi_d", 1.0),
    )


# --- Single-shot MLE ----------------------------------------------------------

def fit_mle(
    dataset: Dataset,
    *,
    n_restarts: int = 5,
    seed: int,
    standard_tcm: bool = False,
    optional_mechanisms: dict[str, bool] | None = None,
    separate_retrieval_weights: bool = False,
) -> dict[str, float]:
    """Return a dict of MLE values for every free parameter (plus derived ones).

    Raises ``FitError`` if no restart converges.
    """
    free_names = _free_parameter_names(
        standard_tcm=standard_tcm,
        optional_mechanisms=optional_mechanisms,
        separate_retrieval_weights=separate_retrieval_weights,
    )
    rng = np.random.default_rng(seed)

    def _neg_log_like(z: np.ndarray) -> float:
        theta = _unpack_theta(
            z, free_names, standard_tcm=standard_tcm,
            separate_retrieval_weights=separate_retrieval_weights,
        )
        try:
            params = _theta_to_params(theta)
        except ValueError:
            return float("inf")
        ll = dataset_log_likelihood(dataset, params)
        return -ll if np.isfinite(ll) else float("inf")

    best_result = None
    best_nll = float("inf")
    for i in range(n_restarts):
        z0 = rng.normal(0.0, 2.0, size=len(free_names))
        try:
            res = minimize(
                _neg_log_like, z0, method="L-BFGS-B", bounds=None,
                options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6},
            )
        except Exception:
            continue
        if res.success and np.isfinite(res.fun) and res.fun < best_nll:
            best_nll = float(res.fun)
            best_result = res

    if best_result is None:
        raise FitError(
            f"No restart converged (n_restarts={n_restarts}). "
            "Try increasing n_restarts or simplifying optional mechanisms."
        )

    return _unpack_theta(
        best_result.x, free_names, standard_tcm=standard_tcm,
        separate_retrieval_weights=separate_retrieval_weights,
    )


# --- FitResult container -------------------------------------------------------

@dataclass(frozen=True)
class FitResult:
    parameters: dict[str, dict[str, float]]
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
    bootstrap_draws: pa.Table

    def save(self, out_dir: str | os.PathLike[str]) -> None:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        payload = {
            "parameters": self.parameters,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "n_participants": self.n_participants,
            "n_lists": self.n_lists,
            "n_recalls_used": self.n_recalls_used,
            "n_intrusions_excluded": self.n_intrusions_excluded,
            "seed": self.seed,
            "elapsed_seconds": self.elapsed_seconds,
            "ms_tcm_version": self.ms_tcm_version,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "standard_tcm": self.standard_tcm,
        }
        write_json_sorted(payload, p / "fit_summary.json")
        pq.write_table(
            self.bootstrap_draws, p / "fit_bootstrap.parquet",
            compression="zstd", compression_level=1,
            use_dictionary=False, write_statistics=False, version="2.6",
        )


def _compute_aic_bic(
    log_likelihood: float, k: int, n_recalls_used: int,
) -> tuple[float, float]:
    aic = 2.0 * k - 2.0 * log_likelihood
    bic = k * float(np.log(max(n_recalls_used, 1))) - 2.0 * log_likelihood
    return aic, bic
