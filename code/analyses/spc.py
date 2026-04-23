"""Serial position curve (SPC): P(recall) by serial position.

Computes SPC from either an empirical ``Dataset`` (uses the
``dataset.recalled`` pyarrow Table) or from a simulated pyarrow Table
matching the ``recalled.parquet`` schema.

The v6 dispatch: ``compute_spc`` accepts either a ``Dataset`` or a
``pa.Table``; both paths return a ``pd.Series`` indexed by serial
position (1..W).

Canonical reference: Howard & Kahana 2002 Figure 1; Kahana et al. 2002
Figure 1; Cornell & Zhang 2025 Figure 2a.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import pyarrow as pa

from ms_tcm.dataset import Dataset


def _spc_from_recalled_df(
    rec_df: pd.DataFrame, groups_df: pd.DataFrame, W: int,
) -> pd.Series:
    """Core SPC computation: for each (participant, list) in ``groups_df``,
    count which serial positions appeared in the recalled table. Divide by
    the number of (participant, list) groups.
    """
    hits = np.zeros(W, dtype=np.float64)
    total = 0
    # Index recalled by (participant, list) for fast lookup.
    in_list = rec_df[rec_df["serial_position"] > 0]
    rec_groups = {
        (int(p), int(l)): set(int(sp) for sp in sub["serial_position"].tolist())
        for (p, l), sub in in_list.groupby(["participant", "list"])
    }
    for _, row in groups_df.iterrows():
        p, l = int(row["participant"]), int(row["list"])
        recalled_sp = rec_groups.get((p, l), set())
        for sp in recalled_sp:
            if 1 <= sp <= W:
                hits[sp - 1] += 1.0
        total += 1
    arr = hits / max(total, 1)
    return pd.Series(arr, index=pd.Index(np.arange(1, W + 1), name="serial_position"),
                     name="p_recall")


def compute_spc(obj: Union[Dataset, pa.Table], W: int | None = None) -> pd.Series:
    """Serial position curve. Dispatches on input type.

    - ``Dataset``: empirical SPC from ``dataset.recalled`` using
      ``dataset.presented`` to enumerate (participant, list) groups.
    - ``pa.Table`` (simulated ``recalled.parquet``): empirical SPC from
      the simulated recalls. ``W`` must be supplied when the table's
      ``serial_position`` max is less than the list length.

    Returns a ``pd.Series`` indexed by serial position (1..W).
    """
    if isinstance(obj, Dataset):
        rec = obj.recalled.to_pandas()
        pres = obj.presented.to_pandas()
        groups = pres[["participant", "list"]].drop_duplicates().reset_index(drop=True)
        W_eff = int(W or obj.num_words_per_list)
        return _spc_from_recalled_df(rec, groups, W_eff)
    elif isinstance(obj, pa.Table):
        rec = obj.to_pandas()
        groups = rec[["participant", "list"]].drop_duplicates().reset_index(drop=True)
        if W is None:
            W = int(max(16, int(rec["serial_position"].max()))) if len(rec) else 16
        return _spc_from_recalled_df(rec, groups, int(W))
    else:
        raise TypeError(
            f"compute_spc accepts Dataset or pa.Table; got {type(obj).__name__!r}"
        )


# --- Backward-compatible aliases (used elsewhere in the code base) ---

def observed(dataset: Dataset) -> np.ndarray:
    """Legacy entry point. Returns the SPC as a numpy array of length W."""
    return compute_spc(dataset).to_numpy()


def predicted_from_samples(synthetic_recalled, W: int) -> np.ndarray:
    """Legacy entry point used by analyses modules.

    Accepts a pa.Table or pandas DataFrame. Returns a numpy array of length W.
    """
    if isinstance(synthetic_recalled, pa.Table):
        return compute_spc(synthetic_recalled, W=W).to_numpy()
    if hasattr(synthetic_recalled, "to_pandas"):
        return compute_spc(synthetic_recalled.to_pandas(), W=W).to_numpy()
    # Treat as DataFrame.
    rec = synthetic_recalled
    groups = rec[["participant", "list"]].drop_duplicates().reset_index(drop=True)
    return _spc_from_recalled_df(rec, groups, W).to_numpy()
