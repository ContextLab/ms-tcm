"""Draw N synthetic datasets from a fit, compute per-draw measures, return bands.

Used by every "observed vs predicted" figure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa

from ms_tcm import Dataset, ModelParameters, MSTCMModel, sample_recalls


def params_from_fit_summary(fit_dir: str | Path) -> ModelParameters:
    """Reconstruct a ``ModelParameters`` from a ``fit_summary.json``."""
    payload = json.loads(Path(fit_dir, "fit_summary.json").read_text())
    p = payload["parameters"]
    def _mle(name, default=None):
        if name in p:
            return float(p[name]["mle"])
        return default
    return ModelParameters(
        beta_global=_mle("beta_global", 0.5),
        beta_storyline=max(_mle("beta_storyline", 0.5), 1e-6),
        w_global=_mle("w_global", 0.5),
        w_storyline=_mle("w_storyline", 1.0 - _mle("w_global", 0.5)),
        w_global_ret=_mle("w_global_ret", None),
        w_storyline_ret=_mle("w_storyline_ret", None),
        gamma=_mle("gamma", 0.0),
        lambda_interference=_mle("lambda_interference", 0.0),
        feature_dim=71,
        seed=0,
    )


def draw_synthetic_datasets(
    dataset: Dataset, parameters: ModelParameters,
    n_draws: int, rng_seed: int,
) -> list[Dataset]:
    """Return ``n_draws`` synthetic datasets (same presented sequences, new recalls)."""
    model = MSTCMModel(parameters)
    master = np.random.SeedSequence(rng_seed)
    children = master.spawn(n_draws)
    out: list[Dataset] = []
    for child in children:
        rng = np.random.default_rng(child)
        synth_rec = sample_recalls(model, dataset, rng)
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
    """Compute median + lower/upper percentile envelope across a set of synthetic
    datasets, per element of the measure's output array.

    Returns (median, ci_lower, ci_upper), each the same shape as the measure.
    Missing/NaN entries in individual draws are handled via nanmedian/nanquantile.
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
