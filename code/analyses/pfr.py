"""Probability of first recall (pFR): P(first recall == serial_position i).

Averaged across (participant, list) groups. Accepts ``Dataset`` (empirical
input) or ``pa.Table`` (simulated recalls).

Canonical reference: Howard & Kahana 2002 Figure 1; Cornell & Zhang 2025
Figure 2b; Kahana et al. 2002 Figure 1.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import pyarrow as pa

from ms_tcm.dataset import Dataset


def _pfr_from_df(rec_df: pd.DataFrame, groups_df: pd.DataFrame, W: int) -> pd.Series:
    first = rec_df[(rec_df["output_position"] == 1) & (rec_df["serial_position"] > 0)]
    # Index first recalls by (participant, list).
    first_by_pl = {
        (int(r["participant"]), int(r["list"])): int(r["serial_position"])
        for _, r in first.iterrows()
    }
    counts = np.zeros(W, dtype=np.float64)
    total = 0
    for _, row in groups_df.iterrows():
        p, l = int(row["participant"]), int(row["list"])
        sp = first_by_pl.get((p, l))
        if sp is not None and 1 <= sp <= W:
            counts[sp - 1] += 1.0
        total += 1
    arr = counts / max(total, 1)
    return pd.Series(
        arr, index=pd.Index(np.arange(1, W + 1), name="serial_position"),
        name="p_first_recall",
    )


def compute_pfr(obj: Union[Dataset, pa.Table], W: int | None = None) -> pd.Series:
    """Probability of first recall. Dispatches on input type.

    Returns a ``pd.Series`` indexed by serial position (1..W).
    """
    if isinstance(obj, Dataset):
        rec = obj.recalled.to_pandas()
        pres = obj.presented.to_pandas()
        groups = pres[["participant", "list"]].drop_duplicates().reset_index(drop=True)
        W_eff = int(W or obj.num_words_per_list)
        return _pfr_from_df(rec, groups, W_eff)
    elif isinstance(obj, pa.Table):
        rec = obj.to_pandas()
        groups = rec[["participant", "list"]].drop_duplicates().reset_index(drop=True)
        if W is None:
            W = int(max(16, int(rec["serial_position"].max()))) if len(rec) else 16
        return _pfr_from_df(rec, groups, int(W))
    else:
        raise TypeError(
            f"compute_pfr accepts Dataset or pa.Table; got {type(obj).__name__!r}"
        )


def observed(dataset: Dataset) -> np.ndarray:
    return compute_pfr(dataset).to_numpy()


def predicted_from_samples(synthetic_recalled, W: int) -> np.ndarray:
    if isinstance(synthetic_recalled, pa.Table):
        return compute_pfr(synthetic_recalled, W=W).to_numpy()
    if hasattr(synthetic_recalled, "to_pandas"):
        return compute_pfr(synthetic_recalled.to_pandas(), W=W).to_numpy()
    rec = synthetic_recalled
    groups = rec[["participant", "list"]].drop_duplicates().reset_index(drop=True)
    return _pfr_from_df(rec, groups, W).to_numpy()
