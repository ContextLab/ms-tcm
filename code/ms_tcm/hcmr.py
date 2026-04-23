"""HierarchicalCMRModel — the v6 MS-TCM orchestrator.

Weaves together drift (``ms_tcm.drift``), boundary transitions
(``ms_tcm.boundaries``), associative matrices (``ms_tcm.matrices``), and the
retrieval route (``ms_tcm.retrieval``). Takes a ``Dataset`` and
``ModelParameters``; emits an ``EncodingState`` with full context
trajectories + M^IC/M^SC matrices per (participant, list), and exposes
``score_first_recall`` / ``score_next_recall`` / ``sample_recalls`` for
downstream use.

v6 encoding flow per item i in a list:

1. Determine flags: event_boundary (e(i) != e(i-1)), storyline_switch
   (s(i) != s(i-1)), storyline_return (s(i) was seen before in this list).
2. Compute c^IN_i per v6 Eq 1.5.1: mixture of M^FC_pre @ f_i and
   M^FC_exp @ f_i weighted by gamma_fc.
3. If storyline_switch: cache outgoing storyline context via
   boundaries.apply_storyline_switch (updates M^SC).
4. If storyline_return: blend with cached via
   boundaries.apply_storyline_return.
5. Drift the active storyline's c^story via drift.update_story_context;
   inactive storylines are carried forward unchanged.
6. Drift c^item via drift.update_item_context.
7. If event_boundary (but no storyline_switch/return already snapped):
   c^item <- c^story via boundaries.apply_event_boundary.
8. Accumulate M^IC (ic_update) and M^FC_exp (fc_exp_update).

In this feature (feature 002) the FRFR-category worked example has one
event per item (every step is an event boundary in the sense that each
study item presentation is its own "event"). We therefore DO NOT treat
each step as firing a within-storyline event boundary; the ``event_id``
column is absent from FRFR-category data, and the encoding collapses the
event-boundary step to an identity. See v6 §2.2 — "within-storyline event
boundaries do not have their own associative matrix", so the only
observable effect is c^item snapping to c^story, which is already implicit
when storyline_switch or storyline_return fires.

For the Xu et al. 2026 cued-recall paradigm (future work), event
boundaries WILL be meaningful and distinct from storyline boundaries; the
orchestrator already handles the distinction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa

from ms_tcm.boundaries import (
    apply_event_boundary,
    apply_storyline_return,
    apply_storyline_switch,
)
from ms_tcm.dataset import Dataset
from ms_tcm.drift import update_item_context, update_story_context
from ms_tcm.features import encode_features
from ms_tcm.matrices import fc_exp_update, ic_update
from ms_tcm.params import ModelParameters
from ms_tcm.preexp import IdentityPreMatrix, MFCPreMatrix
from ms_tcm.retrieval import (
    activation,
    drift_retrieval_context,
    recall_probabilities,
    stopping_probability,
)


def _e_start(d: int) -> np.ndarray:
    """Reserved list-start unit vector (one-hot at index 0)."""
    v = np.zeros(d, dtype=np.float64)
    v[0] = 1.0
    return v


@dataclass(frozen=True)
class EncodingState:
    """Per-list encoding artifacts (c^item trajectory, M^IC, M^SC, etc.)."""

    # Keyed by (participant, list_).
    c_item: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    c_story_per_storyline: dict[tuple[int, int], dict[str, np.ndarray]] = field(default_factory=dict)
    m_ic: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    m_sc: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    active_storyline: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    list_presented: dict[tuple[int, int], pd.DataFrame] = field(default_factory=dict)
    list_features: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    storyline_order: dict[tuple[int, int], list[str]] = field(default_factory=dict)


class HierarchicalCMRModel:
    """Forward simulator + retrieval scorer for the v6 hierarchical CMR."""

    def __init__(
        self,
        parameters: ModelParameters,
        pre_matrix: MFCPreMatrix | None = None,
    ) -> None:
        self.parameters = parameters
        if pre_matrix is None:
            pre_matrix = IdentityPreMatrix(
                n_items=parameters.feature_dim, n_features=parameters.feature_dim,
            )
        self.pre_matrix = pre_matrix

    # --- Encoding ---

    def encode(self, dataset: Dataset) -> EncodingState:
        p = self.parameters
        state = EncodingState()

        pdf = dataset.presented.to_pandas()
        feat_matrix = encode_features(dataset.presented)
        d = feat_matrix.shape[1]
        if p.feature_dim != d:
            raise ValueError(
                f"ModelParameters.feature_dim={p.feature_dim} does not match "
                f"dataset encoded feature dimensionality {d}"
            )

        group_keys = sorted(set(zip(pdf["participant"].tolist(), pdf["list"].tolist())))

        for part, lst in group_keys:
            mask = (pdf["participant"] == part) & (pdf["list"] == lst)
            sub = pdf.loc[mask].sort_values("serial_position").reset_index(drop=False)
            W = len(sub)

            # Enumerate storylines in order of first appearance.
            categories = []
            for cat in sub["category"].tolist():
                if cat not in categories:
                    categories.append(cat)
            n_storylines = len(categories)
            cat_to_idx = {c: i for i, c in enumerate(categories)}

            # Initialize context vectors, M^IC, M^SC, M^FC_exp.
            c_item = _e_start(d)
            c_story_by_cat: dict[str, np.ndarray] = {c: _e_start(d) for c in categories}
            m_ic = np.zeros((d, d), dtype=np.float64)  # n_items == d under IdentityPreMatrix
            m_sc = np.zeros((n_storylines, d), dtype=np.float64) if n_storylines > 0 else np.zeros((1, d))
            m_fc_exp = np.zeros((d, d), dtype=np.float64)

            # Track trajectories for retrieval scoring.
            c_item_traj = np.zeros((W + 1, d), dtype=np.float64)
            c_item_traj[0] = c_item
            active_traj = np.empty(W, dtype=object)
            seen_storylines: set[str] = set()
            prev_cat: str | None = None
            c_story_traj_by_cat: dict[str, np.ndarray] = {
                c: np.zeros((W + 1, d), dtype=np.float64) for c in categories
            }
            for c in categories:
                c_story_traj_by_cat[c][0] = _e_start(d)

            for t in range(W):
                row = sub.iloc[t]
                orig_idx = int(row["index"])
                f_i = feat_matrix[orig_idx]  # pre-normalized unit-norm feature vec
                cat = row["category"]
                cat_idx = cat_to_idx[cat]

                # --- v6 Eq 1.5.1: c^IN = (1-gamma_fc)*M^FC_pre @ f_i + gamma_fc*M^FC_exp @ f_i ---
                # Map each feature vector to its "item index" via argmax of f_i; this
                # gives a canonical item identity for IdentityPreMatrix. For the
                # FRFR-category encoder, f_i is the normalized multi-hot feature
                # vector -- using it directly as c^IN is the simplest choice that
                # matches standard CMR when M^FC_pre = identity. Pre-experimental
                # branch: M^FC_pre.apply(item_indices) -> one-hot -> but our feature
                # vectors are multi-hot, so we simply use f_i as c^IN_pre (per the
                # feature encoder's normalization).
                c_in_pre = f_i  # With the multi-hot normalized encoder, pre = f_i itself.
                c_in_exp = m_fc_exp @ f_i
                c_in = (1.0 - p.gamma_fc) * c_in_pre + p.gamma_fc * c_in_exp
                # Normalize c_in to unit norm so drift preserves ||c|| = 1.
                c_in_norm = float(np.linalg.norm(c_in))
                if c_in_norm > 1e-12:
                    c_in = c_in / c_in_norm
                else:
                    c_in = f_i  # fall back to pre-experimental input

                # --- Determine boundary type ---
                storyline_switch = (prev_cat is not None) and (cat != prev_cat)
                storyline_return = storyline_switch and (cat in seen_storylines)

                # --- Storyline switch / return bookkeeping ---
                if storyline_switch and not p.standard_tcm:
                    outgoing_idx = cat_to_idx[prev_cat]
                    c_story_out = c_story_by_cat[prev_cat]
                    if storyline_return:
                        # First cache the outgoing storyline...
                        _, m_sc = apply_storyline_switch(
                            m_sc, outgoing_idx, c_story_out, c_story_by_cat[cat],
                            n_storylines=n_storylines,
                        )
                        # ...then blend the returning storyline's context with its cache.
                        c_story_by_cat[cat] = apply_storyline_return(
                            m_sc, cat_idx, c_story_by_cat[cat],
                            lambda_reinstate=p.lambda_reinstate,
                            n_storylines=n_storylines,
                        )
                        # Synchronize item to the newly-blended storyline context (v6 Eq 7).
                        c_item = apply_event_boundary(c_item, c_story_by_cat[cat])
                    else:
                        # First-time storyline switch (this storyline not seen before).
                        c_item, m_sc = apply_storyline_switch(
                            m_sc, outgoing_idx, c_story_out, c_story_by_cat[cat],
                            n_storylines=n_storylines,
                        )

                # --- Drift the active storyline (v6 Eq 2) ---
                if not p.standard_tcm:
                    c_story_by_cat[cat] = update_story_context(
                        c_story_by_cat[cat], p.beta_story, c_in,
                    )

                # --- Drift the item-level context (v6 Eq 1) ---
                c_item = update_item_context(c_item, p.beta_enc, c_in)

                # --- Accumulate associative matrices ---
                # For M^IC we use f_i as the item-identity vector (already
                # normalized multi-hot). This is the row-oriented Hebbian update
                # from CMR Eq 2a.
                ic_update(m_ic, c_item, f_i)
                fc_exp_update(m_fc_exp, c_in, f_i)

                # --- Record ---
                c_item_traj[t + 1] = c_item
                for other_cat in categories:
                    c_story_traj_by_cat[other_cat][t + 1] = c_story_by_cat[other_cat]
                active_traj[t] = cat
                seen_storylines.add(cat)
                prev_cat = cat

            key = (int(part), int(lst))
            state.c_item[key] = c_item_traj
            state.c_story_per_storyline[key] = c_story_traj_by_cat
            state.m_ic[key] = m_ic
            state.m_sc[key] = m_sc
            state.active_storyline[key] = active_traj
            state.list_presented[key] = sub.drop(columns=["index"]).reset_index(drop=True)
            state.list_features[key] = np.stack([feat_matrix[int(i)] for i in sub["index"].tolist()])
            state.storyline_order[key] = categories

        return state

    # --- Retrieval ---

    def _list_W(self, state: EncodingState, key: tuple[int, int]) -> int:
        return state.c_item[key].shape[0] - 1

    def score_first_recall(
        self, state: EncodingState, participant: int, list_: int,
    ) -> np.ndarray:
        """Probability distribution over items for the first recall on this list.

        The cue is the end-of-list item-level context (c^item after the last
        item was encoded). Activation is a = (M^IC)^T @ c^item_cue. In free
        recall we restrict the candidate set to items actually studied on
        this list (same participant, same list); the returned vector is of
        length W and indexes the list's items in serial-position order.
        """
        p = self.parameters
        key = (int(participant), int(list_))
        W = self._list_W(state, key)
        c_cue = state.c_item[key][W]  # end-of-list item context
        m_ic = state.m_ic[key]
        features = state.list_features[key]  # (W, d)
        # Activation over all d item identities, then project onto this list's
        # items by dotting with their feature vectors.
        a_all = activation(m_ic, c_cue)  # (d,) — activation per item-identity column
        # Per-list candidate activation: sum over the feature vector's active
        # indices, weighted by each active slot. For the multi-hot encoder
        # with unit-norm features, a_list[j] = features[j] @ a_all.
        a_list = features @ a_all  # (W,)
        return recall_probabilities(a_list, k=p.k)

    def score_next_recall(
        self,
        state: EncodingState,
        participant: int,
        list_: int,
        last_recalled_serial_position: int,
        c_ret: np.ndarray | None = None,
        recalled_sps: set[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Next-recall probabilities after drifting retrieval context toward the last recalled item.

        Returns (probs, c_ret_new) so the caller can chain subsequent calls.
        ``recalled_sps`` is the set of already-recalled 1-based serial
        positions; they are masked to probability 0.
        """
        p = self.parameters
        key = (int(participant), int(list_))
        W = self._list_W(state, key)
        m_ic = state.m_ic[key]
        features = state.list_features[key]  # (W, d)

        if c_ret is None:
            # Initialize c_ret from the end-of-list item context.
            c_ret = state.c_item[key][W].copy()

        # Drift c_ret toward the just-recalled item's c^item (encoded at t = sp).
        sp = int(last_recalled_serial_position)
        if not (1 <= sp <= W):
            raise ValueError(f"last_recalled_serial_position {sp} out of range [1, {W}]")
        c_item_recalled = state.c_item[key][sp]  # sp-th encoded item's c^item
        c_ret_new = drift_retrieval_context(c_ret, p.beta_rec, c_item_recalled)

        a_all = activation(m_ic, c_ret_new)
        a_list = features @ a_all  # (W,)

        if recalled_sps:
            mask = np.ones(W, dtype=bool)
            for s in recalled_sps:
                if 1 <= s <= W:
                    mask[s - 1] = False
            if not mask.any():
                # All items recalled — return a valid but degenerate distribution.
                probs = np.zeros(W)
                probs[sp - 1] = 1.0  # arbitrary; caller should stop before this
                return probs, c_ret_new
            probs = recall_probabilities(a_list, k=p.k, mask=mask)
        else:
            probs = recall_probabilities(a_list, k=p.k)

        return probs, c_ret_new

    def stopping_prob_after_recalls(
        self,
        state: EncodingState,
        participant: int,
        list_: int,
        c_ret: np.ndarray,
        recalled_sps: set[int],
    ) -> float:
        """C&Z Eq 7 stopping probability given the current retrieval context and recalls."""
        p = self.parameters
        key = (int(participant), int(list_))
        W = self._list_W(state, key)
        m_ic = state.m_ic[key]
        features = state.list_features[key]
        a_all = activation(m_ic, c_ret)
        a_list = features @ a_all  # (W,)
        # Split into already-recalled and not-yet-recalled sums.
        mask_r = np.zeros(W, dtype=bool)
        for s in recalled_sps:
            if 1 <= s <= W:
                mask_r[s - 1] = True
        # Use the post-softmax p-weighted activations or the raw summed-activations?
        # C&Z 2025 uses raw summed activations over the exp-scaled competition;
        # we pick a_list (pre-softmax) which is the conventional choice.
        a_r = float(a_list[mask_r].sum()) if mask_r.any() else 0.0
        a_nr = float(a_list[~mask_r].sum()) if (~mask_r).any() else 0.0
        # Ensure non-negative for the ratio (use exponentials of the raw activations
        # so sums stay positive, since a_list may include negatives depending on M^IC).
        # Convention: use exp(k * a_list) as the effective activation weights.
        w = np.exp(p.k * a_list - np.max(p.k * a_list))
        a_r = float(w[mask_r].sum()) if mask_r.any() else 0.0
        a_nr = float(w[~mask_r].sum()) if (~mask_r).any() else 0.0
        return stopping_probability(a_r, a_nr, p.epsilon_d)


def sample_recalls(
    model: HierarchicalCMRModel,
    dataset_skeleton: Dataset,
    rng: np.random.Generator,
    *,
    recall_length_fn: Callable[[int], int] | None = None,
    max_recalls_per_list: int | None = None,
) -> pa.Table:
    """Deterministically sample a ``recalled.parquet``-shaped table from model probabilities.

    Iterates through every (participant, list) in the dataset, runs the
    stopping rule after each recall to decide whether to continue, and
    collects the sampled recalls into a pyarrow Table matching
    ``recalled.parquet`` schema.

    ``recall_length_fn(W) -> R`` overrides the stopping rule (useful for
    tests that want a fixed recall length). ``max_recalls_per_list`` caps
    the total regardless of stopping rule (default: W).
    """
    p = model.parameters
    state = model.encode(dataset_skeleton)
    pdf = dataset_skeleton.presented.to_pandas()
    group_keys = sorted(set(zip(pdf["participant"].tolist(), pdf["list"].tolist())))

    rows: list[dict] = []
    for part, lst in group_keys:
        sub = pdf[(pdf["participant"] == part) & (pdf["list"] == lst)].sort_values("serial_position").reset_index(drop=True)
        W = len(sub)
        cap = max_recalls_per_list if max_recalls_per_list is not None else W
        if recall_length_fn is not None:
            cap = min(cap, int(recall_length_fn(W)))

        list_group = "early" if int(lst) < 8 else "late"

        first_probs = model.score_first_recall(state, int(part), int(lst))
        if not np.isfinite(first_probs).all() or first_probs.sum() <= 0:
            continue
        pick = int(rng.choice(W, p=first_probs / first_probs.sum()))
        picked_sp = int(sub.iloc[pick]["serial_position"])
        recalled_sps: set[int] = {picked_sp}
        rows.append({
            "participant": int(part),
            "list": int(lst),
            "output_position": 1,
            "word": str(sub.iloc[pick]["word"]),
            "category": str(sub.iloc[pick]["category"]),
            "serial_position": picked_sp,
            "list_group": list_group,
        })

        # Initialize c_ret from end-of-list item context.
        key = (int(part), int(lst))
        c_ret = state.c_item[key][-1].copy()

        for out_pos in range(2, cap + 1):
            if len(recalled_sps) >= W:
                break
            if recall_length_fn is None and p.paradigm == "free_recall":
                # Consult the stopping rule.
                p_stop = model.stopping_prob_after_recalls(
                    state, int(part), int(lst), c_ret, recalled_sps,
                )
                if rng.random() < p_stop:
                    break
            next_probs, c_ret = model.score_next_recall(
                state, int(part), int(lst), picked_sp,
                c_ret=c_ret, recalled_sps=recalled_sps,
            )
            total = float(next_probs.sum())
            if total <= 0 or not np.isfinite(total):
                break
            pick = int(rng.choice(W, p=next_probs / total))
            picked_sp = int(sub.iloc[pick]["serial_position"])
            if picked_sp in recalled_sps:
                # Shouldn't happen with masking, but guard.
                break
            recalled_sps.add(picked_sp)
            rows.append({
                "participant": int(part),
                "list": int(lst),
                "output_position": out_pos,
                "word": str(sub.iloc[pick]["word"]),
                "category": str(sub.iloc[pick]["category"]),
                "serial_position": picked_sp,
                "list_group": list_group,
            })

    if not rows:
        return pa.table({
            "participant": pa.array([], type=pa.int64()),
            "list": pa.array([], type=pa.int64()),
            "output_position": pa.array([], type=pa.int64()),
            "word": pa.array([], type=pa.string()),
            "category": pa.array([], type=pa.string()),
            "serial_position": pa.array([], type=pa.int64()),
            "list_group": pa.array([], type=pa.string()),
        })
    return pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
