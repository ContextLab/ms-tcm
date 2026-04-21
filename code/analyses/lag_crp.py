"""Lag-CRP: conditional response probability as a function of lag.

P(recall_next is at serial_position j | just-recalled is at serial_position i)
binned by lag = j - i. H&K 2002; Kahana 1996.
"""

from __future__ import annotations

import numpy as np

from ms_tcm import Dataset


def _crp_from_recalled_df(rdf, W: int) -> np.ndarray:
    """Return lag-CRP over lags -(W-1)..+(W-1); length 2W-1, center at lag=0."""
    lags = np.arange(-(W - 1), W)
    num = np.zeros(len(lags), dtype=np.float64)
    denom = np.zeros(len(lags), dtype=np.float64)
    for (p, l), group in rdf.groupby(["participant", "list"]):
        seq = [
            int(row["serial_position"])
            for _, row in group.sort_values("output_position").iterrows()
            if int(row["serial_position"]) > 0
        ]
        # Deduplicate to match Kahana 1996 convention: only count transitions
        # where the target is not yet recalled.
        seen: set[int] = set()
        prev = None
        for sp in seq:
            if prev is not None and sp not in seen and prev not in (None,):
                # Count the actual transition.
                lag = sp - prev
                idx = lag + (W - 1)
                if 0 <= idx < len(lags):
                    num[idx] += 1.0
                # Count every possible (not-yet-recalled) transition as a denominator.
                for candidate_sp in range(1, W + 1):
                    if candidate_sp == prev:
                        continue
                    if candidate_sp in seen:
                        continue
                    cand_lag = candidate_sp - prev
                    cand_idx = cand_lag + (W - 1)
                    if 0 <= cand_idx < len(lags):
                        denom[cand_idx] += 1.0
            seen.add(sp)
            prev = sp
    with np.errstate(divide="ignore", invalid="ignore"):
        crp = np.where(denom > 0, num / denom, np.nan)
    return crp


def observed(dataset: Dataset) -> np.ndarray:
    W = dataset.num_words_per_list
    rdf = dataset.recalled.to_pandas()
    return _crp_from_recalled_df(rdf, W)


def predicted_from_samples(synthetic_recalled, W: int) -> np.ndarray:
    rdf = synthetic_recalled.to_pandas() if hasattr(synthetic_recalled, "to_pandas") \
        else synthetic_recalled
    return _crp_from_recalled_df(rdf, W)


def lag_axis(W: int) -> np.ndarray:
    return np.arange(-(W - 1), W)
