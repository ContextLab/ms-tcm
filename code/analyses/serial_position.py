"""Serial-position curve: P(recall) by serial position i, ignoring output order."""

from __future__ import annotations

import numpy as np

from ms_tcm import Dataset


def observed(dataset: Dataset) -> np.ndarray:
    W = dataset.num_words_per_list
    pdf = dataset.presented.to_pandas()
    rdf = dataset.recalled.to_pandas()
    # For each (participant, list) count which serial positions were recalled at least once.
    hits = np.zeros(W, dtype=np.float64)
    total = 0
    for (p, l), _ in pdf.groupby(["participant", "list"]):
        r = rdf[(rdf["participant"] == p) & (rdf["list"] == l)]
        recalled_sp = set(int(x) for x in r["serial_position"].tolist() if int(x) > 0)
        for sp in recalled_sp:
            if 1 <= sp <= W:
                hits[sp - 1] += 1.0
        total += 1
    return hits / max(total, 1)


def predicted_from_samples(synthetic_recalled, W: int) -> np.ndarray:
    rdf = synthetic_recalled.to_pandas() if hasattr(synthetic_recalled, "to_pandas") \
        else synthetic_recalled
    hits = np.zeros(W, dtype=np.float64)
    groups = rdf[["participant", "list"]].drop_duplicates()
    total = len(groups)
    for _, row in groups.iterrows():
        p, l = int(row["participant"]), int(row["list"])
        r = rdf[(rdf["participant"] == p) & (rdf["list"] == l)]
        recalled_sp = set(int(x) for x in r["serial_position"].tolist() if int(x) > 0)
        for sp in recalled_sp:
            if 1 <= sp <= W:
                hits[sp - 1] += 1.0
    return hits / max(total, 1)
