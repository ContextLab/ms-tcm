"""Draw N synthetic datasets from a fit, compute per-draw measures, return bands.

Used by every "observed vs predicted" figure. Rewritten for v6 hierarchical
CMR (feature 002-ms-tcm-v6-hcmr); consumes ``HierarchicalCMRModel`` +
``ModelParameters`` via the public ``ms_tcm.hcmr`` surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa

from ms_tcm.dataset import Dataset
from ms_tcm.hcmr import HierarchicalCMRModel, sample_recalls
from ms_tcm.params import ModelParameters


def params_from_fit_summary(fit_dir: str | Path) -> ModelParameters:
    """Reconstruct a v6 ``ModelParameters`` from a ``fit_summary.json``.

    v6 parameter names: beta_enc, beta_story, gamma_fc, k, lambda_reinstate,
    beta_rec, epsilon_d. See specs/002-ms-tcm-v6-hcmr/contracts/model-api.md §5.
    """
    payload = json.loads(Path(fit_dir, "fit_summary.json").read_text())
    p = payload["parameters"]

    def _mle(name, default):
        if name in p:
            return float(p[name]["mle"])
        return default

    kwargs: dict[str, float | bool] = {
        "beta_enc": _mle("beta_enc", 0.679),
        "beta_story": _mle("beta_story", 0.400),
        "gamma_fc": _mle("gamma_fc", 0.315),
        "k": _mle("k", 6.50),
        "lambda_reinstate": _mle("lambda_reinstate", 0.80),
        "beta_rec": _mle("beta_rec", 0.326),
        "epsilon_d": _mle("epsilon_d", 1.04),
    }
    # Clamp beta_story below beta_enc (FR-001 validation).
    if kwargs["beta_story"] >= kwargs["beta_enc"]:
        kwargs["beta_story"] = float(kwargs["beta_enc"]) * 0.5
    if payload.get("standard_tcm", False):
        kwargs["standard_tcm"] = True
    return ModelParameters(**kwargs)


def draw_synthetic_datasets(
    dataset: Dataset, parameters: ModelParameters,
    n_draws: int, rng_seed: int,
) -> list[Dataset]:
    """Return ``n_draws`` synthetic datasets (same presented sequences, new recalls)."""
    model = HierarchicalCMRModel(parameters)
    state = model.encode(dataset)
    master = np.random.SeedSequence(rng_seed)
    children = master.spawn(n_draws)
    out: list[Dataset] = []
    for child in children:
        rng = np.random.default_rng(child)
        synth_rec = sample_recalls(model, dataset, rng, state=state)
        out.append(Dataset(
            presented=dataset.presented, recalled=synth_rec,
            manifest=dict(dataset.manifest),
        ))
    return out


def compute_band(
    synthetic_datasets: Iterable[Dataset],
    measure_fn: Callable[[Dataset], np.ndarray],
    ci: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute median + lower/upper percentile envelope across synthetic datasets.

    Returns (median, ci_lower, ci_upper), each the same shape as the measure.
    """
    stacked = np.stack([np.asarray(measure_fn(ds)) for ds in synthetic_datasets], axis=0)
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    median = np.nanmedian(stacked, axis=0)
    lower = np.nanquantile(stacked, lo_q, axis=0)
    upper = np.nanquantile(stacked, hi_q, axis=0)
    return median, lower, upper


def compute_scalar_band(
    synthetic_datasets: Iterable[Dataset],
    measure_fn: Callable[[Dataset], float],
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Same as compute_band but for scalar-valued measures."""
    vals = np.array([float(measure_fn(ds)) for ds in synthetic_datasets])
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    return (
        float(np.nanmedian(vals)),
        float(np.nanquantile(vals, lo_q)),
        float(np.nanquantile(vals, hi_q)),
    )
