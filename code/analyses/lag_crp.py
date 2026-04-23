"""Lag-conditional response probability (lag-CRP).

P(next recalled item at serial_position j | just-recalled at i), binned by
lag = j - i. Excludes lag 0 (same position, dedup convention). Excludes
transitions where the target has already been recalled (Kahana 1996).

Canonical reference: Kahana 1996; Howard & Kahana 2002; Cornell & Zhang
2025 Figure 2c.

Dispatches on input type: ``Dataset`` (empirical) or ``pa.Table``
(simulated recalls).
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import pyarrow as pa

from ms_tcm.dataset import Dataset


def _crp_from_df(rec_df: pd.DataFrame, W: int) -> pd.Series:
    """Compute lag-CRP over lags in [-(W-1), +(W-1)] excluding 0.

    Returns a ``pd.Series`` indexed by lag (sorted ascending; lag 0 is
    present but NaN so downstream filters ``crp.index != 0`` work).
    """
    lags = np.arange(-(W - 1), W)
    num = np.zeros(len(lags), dtype=np.float64)
    denom = np.zeros(len(lags), dtype=np.float64)
    for (p, l), group in rec_df.groupby(["participant", "list"]):
        seq = [
            int(row["serial_position"])
            for _, row in group.sort_values("output_position").iterrows()
            if int(row["serial_position"]) > 0
        ]
        seen: set[int] = set()
        prev: int | None = None
        for sp in seq:
            if prev is not None and sp not in seen and prev != sp:
                lag = sp - prev
                idx = lag + (W - 1)
                if 0 <= idx < len(lags):
                    num[idx] += 1.0
                for candidate_sp in range(1, W + 1):
                    if candidate_sp == prev or candidate_sp in seen:
                        continue
                    cand_lag = candidate_sp - prev
                    cand_idx = cand_lag + (W - 1)
                    if 0 <= cand_idx < len(lags):
                        denom[cand_idx] += 1.0
            seen.add(sp)
            prev = sp
    with np.errstate(divide="ignore", invalid="ignore"):
        crp = np.where(denom > 0, num / denom, np.nan)
    # Lag 0 is naturally NaN (denom=0) since we never include prev -> prev.
    return pd.Series(crp, index=pd.Index(lags, name="lag"), name="crp")


def compute_lag_crp(obj: Union[Dataset, pa.Table], W: int | None = None) -> pd.Series:
    if isinstance(obj, Dataset):
        rec = obj.recalled.to_pandas()
        W_eff = int(W or obj.num_words_per_list)
        return _crp_from_df(rec, W_eff)
    elif isinstance(obj, pa.Table):
        rec = obj.to_pandas()
        if W is None:
            W = int(max(16, int(rec["serial_position"].max()))) if len(rec) else 16
        return _crp_from_df(rec, int(W))
    else:
        raise TypeError(
            f"compute_lag_crp accepts Dataset or pa.Table; got {type(obj).__name__!r}"
        )


def observed(dataset: Dataset) -> np.ndarray:
    return compute_lag_crp(dataset).to_numpy()


def predicted_from_samples(synthetic_recalled, W: int) -> np.ndarray:
    if isinstance(synthetic_recalled, pa.Table):
        return compute_lag_crp(synthetic_recalled, W=W).to_numpy()
    if hasattr(synthetic_recalled, "to_pandas"):
        return compute_lag_crp(synthetic_recalled.to_pandas(), W=W).to_numpy()
    return _crp_from_df(synthetic_recalled, W).to_numpy()


def lag_axis(W: int) -> np.ndarray:
    return np.arange(-(W - 1), W)
