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
    # Derive w_global from w_storyline when only one is in the summary
    # (which is the case for standard-TCM fits, where w_storyline is fixed
    # at 0 and is therefore stored as "derived" while w_global is implicit).
    w_storyline = _mle("w_storyline", None)
    w_global = _mle("w_global", None)
    if w_global is None and w_storyline is not None:
        w_global = 1.0 - w_storyline
    elif w_global is None and w_storyline is None:
        w_global, w_storyline = 0.5, 0.5
    elif w_storyline is None:
        w_storyline = 1.0 - w_global
    return ModelParameters(
        beta_global=_mle("beta_global", 0.5),
        beta_storyline=max(_mle("beta_storyline", 0.5), 1e-6),
        w_global=w_global,
        w_storyline=w_storyline,
        w_global_ret=_mle("w_global_ret", None),
        w_storyline_ret=_mle("w_storyline_ret", None),
        gamma=_mle("gamma", 0.0),
        lambda_interference=_mle("lambda_interference", 0.0),
        tau=_mle("tau", 1.0),
        phi_s=_mle("phi_s", 0.0),
        phi_d=_mle("phi_d", 1.0),
        feature_dim=71,
        seed=0,
    )


def draw_synthetic_datasets(
    dataset: Dataset, parameters: ModelParameters,
    n_draws: int, rng_seed: int,
) -> list[Dataset]:
    """Return ``n_draws`` synthetic datasets (same presented sequences, new recalls).

    Recall length per synthetic list is set to match the observed dataset's
    mean in-list recall length per (participant, list) pair, rounded to the
    nearest integer. Without this, the model samples W recalls per list (all
    items recalled exactly once under the no-repeats mask), which inflates
    the serial-position curve to a flat 1.0 and erases the primacy/recency
    shape.
    """
    rdf = dataset.recalled.to_pandas()
    in_list = rdf[rdf["serial_position"] > 0]
    mean_recall_len = int(round(
        in_list.groupby(["participant", "list"]).size().mean()
    )) if len(in_list) else dataset.num_words_per_list
    W = dataset.num_words_per_list
    capped_len = max(1, min(mean_recall_len, W))
    model = MSTCMModel(parameters)
    master = np.random.SeedSequence(rng_seed)
    children = master.spawn(n_draws)
    out: list[Dataset] = []
    for child in children:
        rng = np.random.default_rng(child)
        synth_rec = sample_recalls(
            model, dataset, rng,
            recall_length_fn=lambda _W, _n=capped_len: _n,
        )
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
