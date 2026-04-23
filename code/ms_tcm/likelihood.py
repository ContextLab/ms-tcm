"""Per-list and dataset-level log-likelihood for v6 hierarchical CMR.

Observed recall sequence evaluation under the CMR retrieval route:
- First recall: P(first | end-of-list context) via softmax(k * M^IC.T @ c^item_end).
- Subsequent recalls: retrieval context drifts toward last-recalled item at
  rate beta_rec (C&Z Eq 3); next-recall probability masks already-recalled.
- Free-recall stopping: each inter-recall step contributes log(1 - p_stop);
  the act of terminating after the last recall contributes log(p_stop)
  — if there are K observed recalls on a list of W items, the likelihood
  integrates over K-1 "do not stop" decisions plus one "stop" decision.
- Extra-list intrusions (serial_position == 0): excluded from likelihood;
  counted separately in diagnostics.

Spec: specs/002-ms-tcm-v6-hcmr/spec.md FR-005, FR-006.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa

from ms_tcm.dataset import Dataset
from ms_tcm.hcmr import EncodingState, HierarchicalCMRModel
from ms_tcm.params import ModelParameters


# Per-process, per-dataset encoding cache keyed by CONTENT (dataset identity
# AND parameter content), not by ``id()``. Keying by ``id()`` is unsafe because
# Python recycles ``id()`` values after an object is garbage-collected — a
# freshly-allocated ``ModelParameters`` can coincidentally receive the ``id()``
# of a previously-evicted-but-logically-different instance, returning a stale
# EncodingState. That broke determinism in both ``fit_mle`` (same seed two runs
# diverged) and in the serial-vs-parallel bootstrap contract (main process had
# pre-populated cache state, workers did not).
#
# We use ``id(dataset)`` (datasets are not hashable cheaply and, within a
# process, the same object is reused across the whole fit) plus the full frozen
# ``ModelParameters`` instance (hashable by value because the dataclass is
# frozen). Collisions by content are correct; collisions by id recycling are
# impossible here because the key holds a live reference to the params.
_ENCODING_CACHE: "dict[tuple[int, ModelParameters], EncodingState]" = {}
_ENCODING_CACHE_CAP = 64

# Per-dataset cache of the enumerated (participant, list, recalled_df) triples.
# Profiling showed ``Dataset.iter_lists`` was ~43% of total fit wall-clock
# because it did ``presented.to_pandas()`` + pandas boolean indexing +
# ``pa.Table.from_pandas`` for every list on every likelihood evaluation.
# With ~48 lists × 424 optimizer evaluations this is ~20k redundant pandas
# conversions. We cache the per-list slices on first access and return
# (participant, list_, recalled_df) tuples directly; the pandas DataFrame
# form avoids the pyarrow round-trip that ``iter_lists`` otherwise does.
# Keyed by ``id(dataset)`` because the Dataset is a frozen dataclass whose
# Parquet-backed tables are immutable within a process lifetime.
_LIST_CACHE: "dict[int, list[tuple[int, int, pd.DataFrame]]]" = {}
_LIST_CACHE_CAP = 8


def _cached_encode(
    model: HierarchicalCMRModel, dataset: Dataset,
) -> EncodingState:
    """Return a cached EncodingState for (dataset, parameters), encoding if new.

    Keyed by ``(id(dataset), parameters)`` where ``parameters`` is the full
    frozen dataclass (hashed by value). This is safe against ``id()`` reuse
    because the key keeps a live reference to the parameters object, and
    content-keyed so two logically-identical parameter sets share one entry
    (useful e.g. when the optimizer retries an identical theta).
    """
    key = (id(dataset), model.parameters)
    state = _ENCODING_CACHE.get(key)
    if state is None:
        state = model.encode(dataset)
        _ENCODING_CACHE[key] = state
        # Cap the cache at _ENCODING_CACHE_CAP entries to bound memory across
        # many proposals. Python 3.7+ dicts preserve insertion order; evicting
        # ``next(iter(...))`` is FIFO, which is deterministic under identical
        # insertion sequences — the determinism guarantee this function relies
        # on is that identical inputs produce identical encoding states, not
        # that the cache eviction order itself affects the returned value.
        if len(_ENCODING_CACHE) > _ENCODING_CACHE_CAP:
            _ENCODING_CACHE.pop(next(iter(_ENCODING_CACHE)))
    return state


def clear_encoding_cache() -> None:
    """Clear the module-level encoding cache (used by tests to avoid cross-talk)."""
    _ENCODING_CACHE.clear()
    _LIST_CACHE.clear()


def _cached_iter_lists(dataset: Dataset) -> list[tuple[int, int, pd.DataFrame]]:
    """Return a cached list of ``(participant, list_, recalled_df)`` triples.

    Computes the per-list slicing once per Dataset identity and reuses it
    across every likelihood evaluation. Profiling showed this is the single
    largest Tier-1 hot spot: ~43% of fit wall-clock was spent in
    ``Dataset.iter_lists`` doing repeated ``presented.to_pandas()`` +
    pandas boolean indexing + ``pa.Table.from_pandas`` conversions.

    We only need the recalled-DF per list (the presented side is already
    covered by the EncodingState's cached per-list structures). Returning
    pandas DataFrames (not pyarrow Tables) lets ``list_log_likelihood``
    skip the costly pyarrow → pandas conversion that was happening per-call.
    """
    key = id(dataset)
    cached = _LIST_CACHE.get(key)
    if cached is not None:
        return cached

    rdf = dataset.recalled.to_pandas()
    pdf = dataset.presented.to_pandas()
    # Enumerate unique (participant, list_) pairs exactly once.
    keys = sorted(set(zip(pdf["participant"].tolist(), pdf["list"].tolist())))
    # Group recalled rows by (participant, list_) once using pandas groupby;
    # each group becomes a DataFrame we can reuse across fits.
    grouped = rdf.groupby(["participant", "list"], sort=False)
    triples: list[tuple[int, int, pd.DataFrame]] = []
    for p, l in keys:
        if (p, l) in grouped.groups:
            rec_sub = grouped.get_group((p, l)).sort_values("output_position").reset_index(drop=True)
        else:
            rec_sub = rdf.iloc[0:0]  # empty DataFrame with correct schema
        triples.append((int(p), int(l), rec_sub))

    _LIST_CACHE[key] = triples
    # Bound memory across many Dataset instances.
    if len(_LIST_CACHE) > _LIST_CACHE_CAP:
        _LIST_CACHE.pop(next(iter(_LIST_CACHE)))
    return triples


@dataclass(frozen=True)
class LikelihoodDiagnostics:
    n_recalls_used: int
    n_intrusions_excluded: int
    n_lists: int


def list_log_likelihood(
    presented_list: pa.Table,
    recalled_list: pa.Table,
    parameters: ModelParameters,
    encoding_state: EncodingState,
    participant: int,
    list_: int,
) -> float:
    """Log-likelihood of the observed recall sequence for one list."""
    rec_df = recalled_list.to_pandas() if isinstance(recalled_list, pa.Table) else recalled_list
    if len(rec_df) == 0:
        return 0.0

    model = HierarchicalCMRModel(parameters)
    # NB: model.encode produced encoding_state already; we reuse it.
    key = (int(participant), int(list_))
    W = encoding_state.c_item[key].shape[0] - 1

    total = 0.0
    c_ret: np.ndarray | None = None
    recalled_sps: set[int] = set()
    prev_sp: int | None = None
    # Extract the recall column as a numpy int array once — avoids pandas
    # iterrows overhead (another Tier-1 hot path). When the caller is
    # ``dataset_log_likelihood`` the DF is already sorted by
    # ``_cached_iter_lists``; guard with a stable sort otherwise.
    if "output_position" in rec_df.columns:
        if not rec_df["output_position"].is_monotonic_increasing:
            rec_df = rec_df.sort_values("output_position")
    sp_array = rec_df["serial_position"].to_numpy(dtype=np.int64)
    for sp_value in sp_array:
        sp = int(sp_value)
        if sp == 0:
            continue  # extra-list intrusion
        if sp in recalled_sps:
            # Observed repeat; model assigns 0 prob under masking. Treat as
            # a data noise event: skip (do not contribute -inf).
            prev_sp = sp
            continue
        if prev_sp is None:
            probs = model.score_first_recall(encoding_state, participant, list_)
            # c_ret after first recall will be reset to the encoding context
            # of the first recalled item, below (after we score it).
        else:
            if parameters.paradigm == "free_recall":
                # Stopping rule: the model did NOT stop, so include log(1 - p_stop).
                p_stop = model.stopping_prob_after_recalls(
                    encoding_state, participant, list_, c_ret, recalled_sps,
                )
                if p_stop < 1.0:
                    total += float(np.log(1.0 - p_stop))
                else:
                    return float("-inf")
            probs, c_ret = model.score_next_recall(
                encoding_state, participant, list_, prev_sp,
                c_ret=c_ret, recalled_sps=recalled_sps,
            )

        idx = sp - 1
        if not (0 <= idx < W):
            return float("-inf")
        p = float(probs[idx])
        if p <= 0.0 or not np.isfinite(p):
            return float("-inf")
        total += float(np.log(p))
        # After recording the recall at sp, reactivate c_ret to the encoding
        # context of the recalled item (C&Z 2025 Fig 1b) — then apply β_rec
        # drift toward the input e_sp for the NEXT recall's activation.
        c_ret = encoding_state.c_item[key][sp].copy()
        prev_sp = sp
        recalled_sps.add(sp)

    # Account for the final "stop" decision in free recall.
    if parameters.paradigm == "free_recall" and prev_sp is not None and c_ret is not None:
        p_stop = model.stopping_prob_after_recalls(
            encoding_state, participant, list_, c_ret, recalled_sps,
        )
        if 0.0 < p_stop <= 1.0:
            total += float(np.log(p_stop))
        # If p_stop == 0, the model insists on continuing — but the participant
        # stopped. We treat this as a soft signal (don't penalize with -inf);
        # in practice epsilon_d tuning handles this.

    return total


def dataset_log_likelihood(
    dataset: Dataset, parameters: ModelParameters,
) -> float:
    """Sum of list_log_likelihood across every (participant, list).

    Uses two module-level caches for Tier-1 performance:

    1. ``_ENCODING_CACHE`` — skips ``HierarchicalCMRModel.encode`` when the
       optimizer reevaluates the same (dataset, parameters) combination
       (e.g. L-BFGS-B finite-difference probing).
    2. ``_LIST_CACHE`` — skips ``Dataset.iter_lists``' repeated pandas
       ↔ pyarrow conversions. Profiling showed ``iter_lists`` accounted
       for ~43% of fit wall-clock in the pre-cache Tier-1 path; caching
       the per-list recalled-DF slices brings this close to zero.
    """
    model = HierarchicalCMRModel(parameters)
    state = _cached_encode(model, dataset)
    total = 0.0
    for part, lst, rec_sub in _cached_iter_lists(dataset):
        # ``rec_sub`` is already a pandas DataFrame sorted by output_position
        # (see ``_cached_iter_lists``), so ``list_log_likelihood`` skips the
        # pa.Table → DataFrame conversion that was happening per-call.
        contrib = list_log_likelihood(
            None, rec_sub, parameters, state, part, lst,
        )
        if not np.isfinite(contrib):
            return float("-inf")
        total += contrib
    return total


def likelihood_diagnostics(dataset: Dataset) -> LikelihoodDiagnostics:
    rec = dataset.recalled.to_pandas()
    in_list = int((rec["serial_position"] > 0).sum())
    intrusions = int((rec["serial_position"] == 0).sum())
    n_lists = int(
        dataset.presented.to_pandas()[["participant", "list"]].drop_duplicates().shape[0]
    )
    return LikelihoodDiagnostics(
        n_recalls_used=in_list,
        n_intrusions_excluded=intrusions,
        n_lists=n_lists,
    )
