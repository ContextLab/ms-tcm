"""Probability of first recall (PFR): P(output_position == 1 | serial_position = i).

H&K 2002 pg 3; Polyn 2009 used as a canonical measure."""

from __future__ import annotations

import numpy as np

from ms_tcm import Dataset


def observed(dataset: Dataset) -> np.ndarray:
    """Return an array of length W with PFR at each serial position (1..W).

    Averaged across all (participant, list) pairs.
    """
    W = dataset.num_words_per_list
    pdf = dataset.presented.to_pandas()
    rdf = dataset.recalled.to_pandas()
    first = rdf[rdf["output_position"] == 1]
    # Only count first recalls that are actual in-list items.
    first = first[first["serial_position"] > 0]
    counts = np.zeros(W, dtype=np.float64)
    total = 0
    groups = pdf[["participant", "list"]].drop_duplicates()
    for _, row in groups.iterrows():
        p, l = int(row["participant"]), int(row["list"])
        f = first[(first["participant"] == p) & (first["list"] == l)]
        if len(f) > 0:
            sp = int(f.iloc[0]["serial_position"])
            if 1 <= sp <= W:
                counts[sp - 1] += 1.0
        total += 1
    return counts / max(total, 1)


def predicted_from_samples(synthetic_recalled, W: int) -> np.ndarray:
    """Compute PFR from a synthetic recalled.parquet-shaped table."""
    import pandas as pd
    rdf = synthetic_recalled.to_pandas() if hasattr(synthetic_recalled, "to_pandas") \
        else synthetic_recalled
    first = rdf[(rdf["output_position"] == 1) & (rdf["serial_position"] > 0)]
    counts = np.zeros(W, dtype=np.float64)
    groups = rdf[["participant", "list"]].drop_duplicates()
    total = len(groups)
    for _, row in groups.iterrows():
        p, l = int(row["participant"]), int(row["list"])
        f = first[(first["participant"] == p) & (first["list"] == l)]
        if len(f) > 0:
            sp = int(f.iloc[0]["serial_position"])
            if 1 <= sp <= W:
                counts[sp - 1] += 1.0
    return counts / max(total, 1)
