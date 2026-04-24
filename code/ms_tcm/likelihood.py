"""Per-list and dataset-level log-likelihood for v6 hierarchical CMR.

This module is the Tier-1 entry point. All math delegates to the shared
``ms_tcm._likelihood_core`` module (single source of truth; Constitution II);
Tier-1 and Tier-2 (JAX) backends compute identical log-likelihoods from
the same parameters and data. The ``_likelihood_core`` module is driven
here with ``xp=numpy`` and a plain Python ``for`` loop; the JAX backend
drives it with ``xp=jax.numpy`` wrapped in ``jax.lax.scan``.

Observed recall sequence evaluation under the CMR retrieval route:
- First recall: P(first | end-of-list context) via softmax(k * M^IC.T @ c^item_end).
- Subsequent recalls: retrieval context drifts toward last-recalled item at
  rate beta_rec (C&Z Eq 3); next-recall probability masks already-recalled.
- Free-recall stopping: each inter-recall step contributes log(1 - p_stop);
  the act of terminating after the last recall contributes log(p_stop).
- Extra-list intrusions (serial_position == 0): excluded from likelihood;
  counted separately in diagnostics.
- Observed repeats (sp already in recalled_sps): do not contribute to
  the likelihood; prev_sp advances so the next drift targets the repeat's
  sp. This matches the pre-existing Tier-1 behavior (noise-tolerance).

Spec: specs/002-ms-tcm-v6-hcmr/spec.md FR-005, FR-006.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa

from ms_tcm._likelihood_core import compute_list_log_likelihood_numpy
from ms_tcm.dataset import Dataset
from ms_tcm.hcmr import EncodingState, HierarchicalCMRModel
from ms_tcm.params import ModelParameters


# --- module-level caches --------------------------------------------------
#
# The caches below are pure performance (they don't affect semantics).
# ``_ENCODING_CACHE`` avoided re-running ``HierarchicalCMRModel.encode``
# when the optimizer probed the same parameters twice; since the core now
# owns encoding internally, we instead cache the per-list INPUTS
# (cat_indices, W, K) keyed by the Dataset identity so we don't redo
# pandas→numpy conversions on every likelihood evaluation. Encoding still
# runs once per list per parameter proposal, but the encoding math is now
# inside a compiled scan (JAX) or a tight numpy loop (Tier-1).
#
# ``_LIST_CACHE`` caches the per-list recalled-DF slicing, same as before.


@dataclass(frozen=True)
class _ListInputs:
    """Per-list inputs to the core likelihood function.

    ``cat_indices`` is int64 (W,) giving the storyline index per serial
    position. ``W`` is the list length; ``K`` is the number of unique
    storylines in the list.
    """
    cat_indices: np.ndarray  # shape (W,), int64
    W: int
    K: int


_LIST_INPUTS_CACHE: "dict[tuple[int, int, int], _ListInputs]" = {}
_LIST_INPUTS_CACHE_CAP = 2048

_LIST_CACHE: "dict[int, list[tuple[int, int, pd.DataFrame]]]" = {}
_LIST_CACHE_CAP = 8


def clear_encoding_cache() -> None:
    """Clear the module-level caches (used by tests to avoid cross-talk).

    The name is preserved for API compatibility with the pre-refactor
    module; the cache contents have changed (no more EncodingState cache,
    but the semantics — "forget everything" — are unchanged).
    """
    _LIST_INPUTS_CACHE.clear()
    _LIST_CACHE.clear()


def _cached_list_inputs(
    dataset: Dataset, participant: int, list_: int,
) -> _ListInputs:
    """Return cached ``_ListInputs`` for (dataset, participant, list_).

    Keyed by ``(id(dataset), participant, list_)``. The dataset is a frozen
    dataclass so identity implies content equality within a process.
    """
    key = (id(dataset), int(participant), int(list_))
    cached = _LIST_INPUTS_CACHE.get(key)
    if cached is not None:
        return cached

    pdf = dataset.presented.to_pandas()
    mask = (pdf["participant"] == participant) & (pdf["list"] == list_)
    sub = pdf.loc[mask].sort_values("serial_position").reset_index(drop=True)
    W = len(sub)
    cats: list[str] = []
    for cat in sub["category"].tolist():
        if cat not in cats:
            cats.append(cat)
    K = max(1, len(cats))
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    cat_indices = np.array(
        [cat_to_idx[c] for c in sub["category"].tolist()], dtype=np.int64,
    )
    inputs = _ListInputs(cat_indices=cat_indices, W=W, K=K)
    _LIST_INPUTS_CACHE[key] = inputs
    if len(_LIST_INPUTS_CACHE) > _LIST_INPUTS_CACHE_CAP:
        _LIST_INPUTS_CACHE.pop(next(iter(_LIST_INPUTS_CACHE)))
    return inputs


def _cached_iter_lists(dataset: Dataset) -> list[tuple[int, int, pd.DataFrame]]:
    """Return a cached list of ``(participant, list_, recalled_df)`` triples."""
    key = id(dataset)
    cached = _LIST_CACHE.get(key)
    if cached is not None:
        return cached

    rdf = dataset.recalled.to_pandas()
    pdf = dataset.presented.to_pandas()
    keys = sorted(set(zip(pdf["participant"].tolist(), pdf["list"].tolist())))
    grouped = rdf.groupby(["participant", "list"], sort=False)
    triples: list[tuple[int, int, pd.DataFrame]] = []
    for p, l in keys:
        if (p, l) in grouped.groups:
            rec_sub = grouped.get_group((p, l)).sort_values("output_position").reset_index(drop=True)
        else:
            rec_sub = rdf.iloc[0:0]
        triples.append((int(p), int(l), rec_sub))

    _LIST_CACHE[key] = triples
    if len(_LIST_CACHE) > _LIST_CACHE_CAP:
        _LIST_CACHE.pop(next(iter(_LIST_CACHE)))
    return triples


@dataclass(frozen=True)
class LikelihoodDiagnostics:
    n_recalls_used: int
    n_intrusions_excluded: int
    n_lists: int


def _build_recall_arrays(
    rec_df: pd.DataFrame, W: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (recall_sps, recall_mask) padded to max(W, n_recalls).

    Pads to max(W, n_recalls) so lists where participants produced more
    recalls than there are items in the list (many intra-recall repeats)
    are not silently truncated. The padding length only affects the size
    of the numpy arrays; it does NOT affect the computed LL (padding rows
    have valid=False).
    """
    if "output_position" in rec_df.columns:
        if not rec_df["output_position"].is_monotonic_increasing:
            rec_df = rec_df.sort_values("output_position")
    sps_all = rec_df["serial_position"].to_numpy(dtype=np.int64)
    # Drop extra-list intrusions (sp == 0).
    sps = sps_all[sps_all > 0]
    n = len(sps)
    R = max(W, n, 1)
    recall_sps = np.zeros(R, dtype=np.int64)
    recall_mask = np.zeros(R, dtype=bool)
    recall_sps[:n] = sps
    recall_mask[:n] = True
    return recall_sps, recall_mask


def list_log_likelihood(
    presented_list: pa.Table | None,
    recalled_list: pa.Table | pd.DataFrame,
    parameters: ModelParameters,
    encoding_state: EncodingState | None,
    participant: int,
    list_: int,
    *,
    dataset: Dataset | None = None,
) -> float:
    """Log-likelihood of the observed recall sequence for one list.

    Delegates to ``ms_tcm._likelihood_core.compute_list_log_likelihood_numpy``.
    The ``encoding_state`` parameter is accepted for backwards compatibility
    but is not consumed — the core encodes internally. The ``dataset``
    keyword-only argument is preferred for the per-list-inputs cache;
    callers that don't pass it pay a small re-slicing cost.

    Parameters
    ----------
    presented_list, recalled_list:
        Per-list pyarrow tables (or pandas DataFrame for ``recalled_list``).
        ``presented_list`` is accepted for backwards-compat and is not used
        when ``dataset`` is provided.
    parameters:
        The MS-TCM v6 parameters.
    encoding_state:
        Deprecated; accepted for backwards-compat only.
    participant, list_:
        Identify the (participant, list) pair.
    dataset:
        The full dataset (optional). If provided, the per-list-inputs cache
        keyed on ``id(dataset)`` is used, making repeated calls for the
        same list cheap.
    """
    rec_df = (
        recalled_list.to_pandas()
        if isinstance(recalled_list, pa.Table)
        else recalled_list
    )
    if len(rec_df) == 0:
        return 0.0

    if dataset is not None:
        inputs = _cached_list_inputs(dataset, participant, list_)
    elif presented_list is not None:
        sub = (
            presented_list.to_pandas()
            .sort_values("serial_position")
            .reset_index(drop=True)
        )
        W = len(sub)
        cats: list[str] = []
        for cat in sub["category"].tolist():
            if cat not in cats:
                cats.append(cat)
        K = max(1, len(cats))
        cat_to_idx = {c: i for i, c in enumerate(cats)}
        cat_indices = np.array(
            [cat_to_idx[c] for c in sub["category"].tolist()], dtype=np.int64,
        )
        inputs = _ListInputs(cat_indices=cat_indices, W=W, K=K)
    else:
        raise ValueError(
            "list_log_likelihood requires either presented_list or dataset"
        )

    recall_sps, recall_mask = _build_recall_arrays(rec_df, inputs.W)

    return compute_list_log_likelihood_numpy(
        parameters,
        inputs.cat_indices,
        recall_sps,
        recall_mask,
        W=inputs.W,
        K=inputs.K,
    )


def dataset_log_likelihood(
    dataset: Dataset, parameters: ModelParameters,
) -> float:
    """Sum of ``list_log_likelihood`` across every (participant, list).

    Uses ``_cached_iter_lists`` to skip redundant pandas ↔ pyarrow
    conversions across optimizer iterations; per-list inputs (cat_indices)
    are cached by ``_cached_list_inputs``.
    """
    total = 0.0
    for part, lst, rec_sub in _cached_iter_lists(dataset):
        contrib = list_log_likelihood(
            None, rec_sub, parameters, None, part, lst, dataset=dataset,
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
