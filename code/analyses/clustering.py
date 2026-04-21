"""Clustering scores: temporal, category, semantic.

Follows Polyn et al. 2009's percentile-rank clustering-score formulation.
For each successive recall transition (r_{k-1} -> r_k), rank the not-yet-
recalled items by their distance (in the relevant feature space) to r_{k-1};
compute the percentile rank of r_k in that ordering. Average across
transitions.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ms_tcm import Dataset


def _percentile_rank_clustering(
    rdf: pd.DataFrame,
    pdf: pd.DataFrame,
    distance_fn: Callable[[pd.Series, pd.DataFrame], np.ndarray],
) -> float:
    """Polyn 2009 percentile-rank clustering.

    For each transition where the next recall is in-list and not-yet-recalled,
    compute the fraction of not-yet-recalled candidates whose distance to the
    just-recalled item is greater than the actual next recall's distance.
    Score 0.5 = no clustering; 1.0 = perfect clustering.
    """
    scores: list[float] = []
    for (p, l), rec_group in rdf.groupby(["participant", "list"]):
        pres = pdf[(pdf["participant"] == p) & (pdf["list"] == l)]
        if len(pres) == 0:
            continue
        seq = [
            int(r["serial_position"])
            for _, r in rec_group.sort_values("output_position").iterrows()
            if int(r["serial_position"]) > 0
        ]
        seen: set[int] = set()
        prev_sp = None
        for sp in seq:
            if prev_sp is not None and sp not in seen:
                remaining = pres[~pres["serial_position"].isin(seen | {prev_sp})]
                if len(remaining) > 1 and sp in remaining["serial_position"].values:
                    prev_row = pres[pres["serial_position"] == prev_sp].iloc[0]
                    d = distance_fn(prev_row, remaining)
                    target_d = float(d[remaining["serial_position"].values == sp][0])
                    # Percentile: fraction with STRICTLY greater distance plus
                    # half of ties at this distance.
                    strictly_greater = float((d > target_d).sum())
                    ties = float((d == target_d).sum()) - 1  # exclude the target
                    n_rem = len(d)
                    scores.append((strictly_greater + 0.5 * ties) / (n_rem - 1))
            seen.add(sp)
            prev_sp = sp
    return float(np.mean(scores)) if scores else float("nan")


def _lag_distance(prev_row: pd.Series, remaining: pd.DataFrame) -> np.ndarray:
    return np.abs(remaining["serial_position"].values - int(prev_row["serial_position"]))


def _category_distance(prev_row: pd.Series, remaining: pd.DataFrame) -> np.ndarray:
    same = remaining["category"].values == prev_row["category"]
    return (~same).astype(np.float64)


def _semantic_distance(
    embeddings: dict[str, np.ndarray],
) -> Callable[[pd.Series, pd.DataFrame], np.ndarray]:
    def _inner(prev_row: pd.Series, remaining: pd.DataFrame) -> np.ndarray:
        prev_emb = embeddings[str(prev_row["word"])]
        dists = np.empty(len(remaining), dtype=np.float64)
        for i, (_, rem_row) in enumerate(remaining.iterrows()):
            emb = embeddings[str(rem_row["word"])]
            num = float(np.dot(prev_emb, emb))
            den = float(np.linalg.norm(prev_emb) * np.linalg.norm(emb))
            if den == 0.0:
                dists[i] = 1.0
            else:
                dists[i] = 1.0 - (num / den)   # cosine distance
        return dists
    return _inner


def observed_temporal(dataset: Dataset) -> float:
    rdf = dataset.recalled.to_pandas()
    pdf = dataset.presented.to_pandas()
    return _percentile_rank_clustering(rdf, pdf, _lag_distance)


def observed_category(dataset: Dataset) -> float:
    rdf = dataset.recalled.to_pandas()
    pdf = dataset.presented.to_pandas()
    return _percentile_rank_clustering(rdf, pdf, _category_distance)


def observed_semantic(
    dataset: Dataset, embeddings: dict[str, np.ndarray],
) -> float:
    rdf = dataset.recalled.to_pandas()
    pdf = dataset.presented.to_pandas()
    return _percentile_rank_clustering(rdf, pdf, _semantic_distance(embeddings))


def predicted_temporal_from_samples(
    synthetic_recalled, synthetic_presented,
) -> float:
    rdf = synthetic_recalled.to_pandas() if hasattr(synthetic_recalled, "to_pandas") \
        else synthetic_recalled
    pdf = synthetic_presented.to_pandas() if hasattr(synthetic_presented, "to_pandas") \
        else synthetic_presented
    return _percentile_rank_clustering(rdf, pdf, _lag_distance)


def predicted_category_from_samples(
    synthetic_recalled, synthetic_presented,
) -> float:
    rdf = synthetic_recalled.to_pandas() if hasattr(synthetic_recalled, "to_pandas") \
        else synthetic_recalled
    pdf = synthetic_presented.to_pandas() if hasattr(synthetic_presented, "to_pandas") \
        else synthetic_presented
    return _percentile_rank_clustering(rdf, pdf, _category_distance)


def predicted_semantic_from_samples(
    synthetic_recalled, synthetic_presented,
    embeddings: dict[str, np.ndarray],
) -> float:
    rdf = synthetic_recalled.to_pandas() if hasattr(synthetic_recalled, "to_pandas") \
        else synthetic_recalled
    pdf = synthetic_presented.to_pandas() if hasattr(synthetic_presented, "to_pandas") \
        else synthetic_presented
    return _percentile_rank_clustering(rdf, pdf, _semantic_distance(embeddings))
