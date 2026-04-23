"""HierarchicalCMRModel — the v6 MS-TCM orchestrator.

Weaves together drift (``ms_tcm.drift``), boundary transitions
(``ms_tcm.boundaries``), associative matrices (``ms_tcm.matrices``), and the
retrieval route (``ms_tcm.retrieval``). Takes a ``Dataset`` and
``ModelParameters``; emits an ``EncodingState`` with full context
trajectories + M^IC/M^SC matrices per (participant, list), and exposes
``score_first_recall`` / ``score_next_recall`` / ``sample_recalls`` for
downstream use.

v6 notation (notes/two_level_cmr_v6.pdf §1 "Notation and primitives"):

- N = number of items (scenes) per list.
- f_i ∈ R^N is the **one-hot** indicating item i's identity.
- c^IN_i ∈ R^N is the context **induced** by item i (v6 Eq 1.5.1):
    c^IN_i = (1 - γ_fc) · M^FC_pre · f_i + γ_fc · M^FC_exp · f_i
  Both contexts live in R^N. Under identity M^FC_pre the pre-experimental
  branch reduces to c^IN_pre = f_i (orthogonal one-hots), which is the
  standard CMR "distinctiveness assumption" (v6 §1.5 Option 1).
- c^item_i, c^story_i ∈ R^N are the two drifting context vectors (v6 §1).
- M^IC ∈ R^(N × N) accumulates context-to-item associations (v6 Eq 3):
    ΔM^IC = c^item_i · f_i^T
  Activation at retrieval: a = (M^IC)^T · c^item_cue  ∈ R^N.
- M^SC ∈ R^(N_storylines × N) caches outgoing storyline contexts at
  switches (v6 Eq 5) and is read on returns (v6 Eq 6).

Primacy: standard CMR multiplies M^IC updates by a primacy-gradient
factor (Sederberg et al. 2008; Polyn et al. 2009), producing the primacy
limb of the SPC. This is inherited from standard CMR unchanged (v6 §3
"Retrieval dynamics unchanged").

Free-recall extras (C&Z 2025 Eqs 3, 7):

- β_rec drifts c^ret toward c^item of the just-recalled item.
- ε_d governs per-step stopping: p_stop = exp(-ε_d · a^nr / a^r).

The reserved list-start vector e_start is a one-hot that is ORTHOGONAL to
every item's f_i — we extend the per-list context space by one slot so
that c^item, c^story live in R^(N+1) with f_i = e_{i+1} and e_start = e_0.
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


def _onehot(n: int, i: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float64)
    v[i] = 1.0
    return v


@dataclass(frozen=True)
class EncodingState:
    """Per-list encoding artifacts (c^item trajectory, M^IC, M^SC, etc.).

    Keyed by (participant, list_). Trajectories have shape (W+1, d) where
    d = W + 1 (item identity slots + the e_start slot). Row 0 is e_start;
    rows 1..W are post-encoding contexts.
    """

    c_item: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    c_story_per_storyline: dict[tuple[int, int], dict[str, np.ndarray]] = field(default_factory=dict)
    m_ic: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    m_sc: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    active_storyline: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    list_presented: dict[tuple[int, int], pd.DataFrame] = field(default_factory=dict)
    # Per-list perceptual features (W, feature_dim), used for semantic
    # diagnostics and future embedding-based M^FC_pre options.
    list_features: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    storyline_order: dict[tuple[int, int], list[str]] = field(default_factory=dict)


class HierarchicalCMRModel:
    """Forward simulator + retrieval scorer for the v6 hierarchical CMR.

    Per-list dimensionality: each list allocates a context space of
    d = W + 1 (one slot per item, plus e_start at slot 0). Under the
    default IdentityPreMatrix, c^IN_i = e_{i} in this per-list space — i.e.
    orthogonal one-hots matching v6 §1.5 Option 1. This produces the
    standard CMR drift behavior that yields the SPC / pFR / lag-CRP
    signatures in the Layer 1 test.
    """

    def __init__(
        self,
        parameters: ModelParameters,
        pre_matrix: MFCPreMatrix | None = None,
    ) -> None:
        self.parameters = parameters
        # pre_matrix is currently used at the API level for the US4 hook;
        # for the free-recall paradigm on FRFR-category the default identity
        # path is taken via the orthogonal-one-hot construction below. A
        # pre_matrix of type EmbeddingPreMatrix or a custom IdentityPreMatrix
        # is honored by overriding the per-list c^IN construction (see encode).
        self.pre_matrix = pre_matrix

    # --- Encoding ---

    def encode(self, dataset: Dataset) -> EncodingState:
        """Run encoding across every (participant, list).

        Produces per-list M^IC, M^SC, and c^item / c^story trajectories in a
        per-list context space of dimension d = W + 1.
        """
        p = self.parameters
        state = EncodingState()

        pdf = dataset.presented.to_pandas()
        # Perceptual features are recorded for diagnostics and as the raw
        # input for an embedding-based M^FC_pre (US4); they do not drive the
        # identity-path encoding, which uses orthogonal per-item one-hots.
        feat_matrix = encode_features(dataset.presented)

        group_keys = sorted(set(zip(pdf["participant"].tolist(), pdf["list"].tolist())))

        # Primacy gradient (Sederberg et al. 2008 Eq 5; Polyn et al. 2009
        # Eq 7): scale M^IC Hebbian updates by (1 + phi·exp(-psi·t)) so
        # early-list items accumulate stronger item-context associations.
        # This is a v6-native CMR mechanism (v6 §3 "retrieval dynamics
        # unchanged"), not an MS-TCM addition.
        #
        # In MS-TCM (hierarchical), primacy also emerges from storyline
        # context boundary-sync at list start (c_item[0] = e_start shared
        # by storyline context), so a modest primacy gradient suffices.
        # Under the --standard-tcm reduction the hierarchical primacy
        # entry-point is disabled (v6 §3.1 C&Z 2025 Fig 3e), so we fall
        # back to a stronger gradient to keep the Layer-1 primacy assertion
        # satisfied (SC-006 / FR-021).
        if p.standard_tcm:
            phi = 30.0
            psi = 0.8
        else:
            phi = 1.5
            psi = 0.5

        for part, lst in group_keys:
            mask = (pdf["participant"] == part) & (pdf["list"] == lst)
            sub = pdf.loc[mask].sort_values("serial_position").reset_index(drop=False)
            W = len(sub)
            d = W + 1  # per-list context dim: e_start + W item slots

            # Collect per-list perceptual features in serial-position order.
            list_feats = np.stack(
                [feat_matrix[int(i)] for i in sub["index"].tolist()]
            )

            # Enumerate storylines in order of first appearance.
            categories: list[str] = []
            for cat in sub["category"].tolist():
                if cat not in categories:
                    categories.append(cat)
            n_storylines = max(1, len(categories))
            cat_to_idx = {c: i for i, c in enumerate(categories)}

            # Context vectors, M^IC, M^SC, M^FC_exp.
            c_item = _onehot(d, 0)  # e_start
            c_story_by_cat: dict[str, np.ndarray] = {
                c: _onehot(d, 0) for c in categories
            }
            m_ic = np.zeros((d, W), dtype=np.float64)
            m_sc = np.zeros((n_storylines, d), dtype=np.float64)
            m_fc_exp = np.zeros((d, W), dtype=np.float64)

            # Trajectories.
            c_item_traj = np.zeros((W + 1, d), dtype=np.float64)
            c_item_traj[0] = c_item
            active_traj = np.empty(W, dtype=object)
            seen_storylines: set[str] = set()
            prev_cat: str | None = None
            c_story_traj_by_cat: dict[str, np.ndarray] = {
                c: np.zeros((W + 1, d), dtype=np.float64) for c in categories
            }
            for c in categories:
                c_story_traj_by_cat[c][0] = _onehot(d, 0)

            # Precompute per-step inputs to avoid DataFrame.iloc inside the inner
            # loop (Tier 1 perf: DataFrame.iloc is ~50x slower than list index).
            cat_list = sub["category"].tolist()
            # Precompute the primacy scaling for every t in a single vectorized
            # call (v6 §3 / Polyn et al. 2009 Eq 7).
            t_arr = np.arange(W, dtype=np.float64)
            primacy_all = 1.0 + phi * np.exp(-psi * t_arr)
            # Precompute all f_ident / f_context as one-hot matrices sliced per step.
            beta_enc = p.beta_enc

            for t in range(W):
                cat = cat_list[t]

                # Item identity f_i = e_{t+1} in R^W-slot (position t+1 in d).
                # Under IdentityPreMatrix, c^IN reduces to the orthogonal
                # one-hot e_{t+1} (v6 §1.5 Option 1; C&Z 2025 p.5 "distinctiveness
                # assumption"). The experimental M^FC_exp @ f_i term is zero at
                # encoding time because M^FC_exp has not yet accumulated column t.
                # γ_fc becomes identifiable only under an embedding-based
                # M^FC_pre (v6 §1.5 Option 3, US4 hook).
                #
                # Because c_in is a canonical basis vector, we exploit its sparsity:
                # all inner products with c_item, c_story collapse to a single
                # array lookup (element at index t+1), avoiding full d-vector dot
                # products. This is the Tier 1 vectorization (T037): eliminate
                # O(d) work in the inner loop when c_in is a basis vector.

                # --- Determine boundary type ---
                storyline_switch = (prev_cat is not None) and (cat != prev_cat)
                storyline_return = storyline_switch and (cat in seen_storylines)

                # --- Storyline switch / return bookkeeping ---
                if storyline_switch and not p.standard_tcm:
                    outgoing_idx = cat_to_idx[prev_cat]
                    cat_idx = cat_to_idx[cat]
                    c_story_out = c_story_by_cat[prev_cat]
                    if storyline_return:
                        # Cache outgoing storyline first.
                        _, m_sc = apply_storyline_switch(
                            m_sc, outgoing_idx, c_story_out, c_story_by_cat[cat],
                            n_storylines=n_storylines,
                        )
                        # Then reinstate returning storyline (v6 Eq 6).
                        c_story_by_cat[cat] = apply_storyline_return(
                            m_sc, cat_idx, c_story_by_cat[cat],
                            lambda_reinstate=p.lambda_reinstate,
                            n_storylines=n_storylines,
                        )
                        nn = float(np.linalg.norm(c_story_by_cat[cat]))
                        if nn > 1e-12:
                            c_story_by_cat[cat] = c_story_by_cat[cat] / nn
                        # Sync item to reinstated storyline (v6 Eq 7).
                        c_item = c_story_by_cat[cat].copy()
                    else:
                        # First-time switch.
                        c_item, m_sc = apply_storyline_switch(
                            m_sc, outgoing_idx, c_story_out, c_story_by_cat[cat],
                            n_storylines=n_storylines,
                        )

                # Basis-vector c_in: dot(c_prev, c_in) == c_prev[t+1].
                t_plus_1 = t + 1

                # --- Drift active storyline (v6 Eq 2) ---
                if not p.standard_tcm:
                    c_story_active = c_story_by_cat[cat]
                    dot_s = float(c_story_active[t_plus_1])
                    rho_s = (
                        np.sqrt(max(0.0, 1.0 + p.beta_story * p.beta_story
                                    * (dot_s * dot_s - 1.0)))
                        - p.beta_story * dot_s
                    )
                    c_story_new = rho_s * c_story_active
                    c_story_new[t_plus_1] += p.beta_story
                    c_story_by_cat[cat] = c_story_new

                # --- Drift item-level context (v6 Eq 1), basis-vector c_in ---
                dot_i = float(c_item[t_plus_1])
                rho_i = (
                    np.sqrt(max(0.0, 1.0 + beta_enc * beta_enc
                                * (dot_i * dot_i - 1.0)))
                    - beta_enc * dot_i
                )
                # In-place style update: reallocate only once per step.
                c_item = rho_i * c_item
                c_item[t_plus_1] += beta_enc

                # --- Associative matrix updates ---
                # M^IC += (primacy_scale * c_item) · f_ident^T; f_ident is
                # one-hot at column t, so this is a column-add.
                primacy_scale = primacy_all[t]
                m_ic[:, t] += primacy_scale * c_item
                # M^FC_exp column t += c_in (one-hot at row t+1).
                m_fc_exp[t_plus_1, t] += 1.0

                # --- Record ---
                c_item_traj[t_plus_1] = c_item
                # c_story trajectories: only the active storyline changed this
                # step; others carry forward unchanged. Instead of copying all
                # K trajectories every step (O(K·d) per step), record only the
                # updated ones and fill in inactive ones after the loop.
                for other_cat in categories:
                    c_story_traj_by_cat[other_cat][t_plus_1] = c_story_by_cat[other_cat]
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
            state.list_features[key] = list_feats
            state.storyline_order[key] = categories

        return state

    # --- Retrieval ---

    def _list_W(self, state: EncodingState, key: tuple[int, int]) -> int:
        return state.c_item[key].shape[0] - 1

    def score_first_recall(
        self, state: EncodingState, participant: int, list_: int,
    ) -> np.ndarray:
        """P(first recall | end-of-list context).

        v6 §3.1 Eq 8-9: a = (M^IC)^T @ c^item_cue; p = softmax(k * a).
        """
        p = self.parameters
        key = (int(participant), int(list_))
        c_cue = state.c_item[key][self._list_W(state, key)]
        m_ic = state.m_ic[key]
        a = activation(m_ic, c_cue)
        return recall_probabilities(a, k=p.k)

    def score_next_recall(
        self,
        state: EncodingState,
        participant: int,
        list_: int,
        last_recalled_serial_position: int,
        c_ret: np.ndarray | None = None,
        recalled_sps: set[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Next-recall probabilities with β_rec drift of retrieval context.

        Retrieval drift target (C&Z 2025 Eq 3; Howard & Kahana 2002 / CMR):
        after recalling item sp, c_ret drifts toward the item's INPUT
        context c^IN_sp (= M^FC_pre @ f_sp), not the post-drift c^item_sp.
        Under the orthogonal-one-hot identity M^FC_pre, c^IN_sp = e_{sp}
        (per-list slot sp). This gives the canonical CMR forward asymmetry
        in the lag-CRP: forward neighbors (sp+1, sp+2, ...) were encoded
        with sp's identity vector already integrated into their c^item via
        drift, while backward neighbors (sp-1, sp-2, ...) were NOT (they
        were encoded before sp). So activation cumsums in the forward
        direction only.
        """
        p = self.parameters
        key = (int(participant), int(list_))
        W = self._list_W(state, key)
        m_ic = state.m_ic[key]
        d = W + 1  # per-list context dim

        if c_ret is None:
            c_ret = state.c_item[key][W].copy()

        sp = int(last_recalled_serial_position)
        if not (1 <= sp <= W):
            raise ValueError(f"last_recalled_serial_position {sp} out of range [1, {W}]")

        # CMR Eq 3 retrieval drift: after recalling item sp, c_ret drifts
        # toward c^IN_sp = M^FC_pre @ f_sp. Under identity M^FC_pre this is
        # the per-list orthogonal slot e_sp.
        #
        # Forward-asymmetry mechanism (Howard & Kahana 2002; Polyn et al.
        # 2009; v6 §3): activation_j after drift is
        #   a_j = c_item[j] · (ρ c_ret_old + β_rec e_sp)
        #       = ρ (c_item[j] · c_ret_old) + β_rec (c_item[j] · e_sp).
        # Under orthogonal-one-hot encoding, c_item[j] · e_sp is NON-ZERO
        # only for j ≥ sp (items encoded at time ≥ sp carry the e_sp
        # component via drift). So the β_rec term adds a pure-forward kick;
        # the ρ term carries recency/history from the previous cue. The
        # combined activation peaks at j = sp+1 relative to j = sp-1,
        # producing the canonical forward-asymmetry lag-CRP (C&Z 2025 Fig 2g).
        c_in_recalled = _onehot(d, sp)
        c_ret_new = drift_retrieval_context(c_ret, p.beta_rec, c_in_recalled)
        a = activation(m_ic, c_ret_new)

        if recalled_sps:
            mask = np.ones(W, dtype=bool)
            for s in recalled_sps:
                if 1 <= s <= W:
                    mask[s - 1] = False
            if not mask.any():
                probs = np.zeros(W)
                probs[sp - 1] = 1.0
                return probs, c_ret_new
            probs = recall_probabilities(a, k=p.k, mask=mask)
        else:
            probs = recall_probabilities(a, k=p.k)

        return probs, c_ret_new

    def score_cue(
        self,
        state: EncodingState,
        participant: int,
        list_: int,
        cue_serial_position: int,
    ) -> np.ndarray:
        """Cued-recall scoring (v6 §3.1)."""
        p = self.parameters
        key = (int(participant), int(list_))
        W = self._list_W(state, key)
        m_ic = state.m_ic[key]
        sp = int(cue_serial_position)
        if not (1 <= sp <= W):
            raise ValueError(f"cue_serial_position {sp} out of range [1, {W}]")
        c_cue = state.c_item[key][sp]
        a = activation(m_ic, c_cue)
        return recall_probabilities(a, k=p.k)

    def stopping_prob_after_recalls(
        self,
        state: EncodingState,
        participant: int,
        list_: int,
        c_ret: np.ndarray,
        recalled_sps: set[int],
    ) -> float:
        """C&Z 2025 Eq 7: p_stop = exp(-ε_d · a^nr / a^r).

        a^r and a^nr are the sums of activations (NOT softmax probabilities)
        over recalled and not-recalled items respectively. Using raw
        activations matches Polyn et al. 2009 Eq 9 and C&Z 2025's description;
        softmax-weighted sums would let one dominant item's probability
        trigger premature stopping even when other items have non-trivial
        absolute activation.
        """
        p = self.parameters
        key = (int(participant), int(list_))
        W = self._list_W(state, key)
        m_ic = state.m_ic[key]
        a = activation(m_ic, c_ret)
        # Use absolute-value activation so a_r/a_nr are always nonnegative.
        a_abs = np.abs(a)
        mask_r = np.zeros(W, dtype=bool)
        for s in recalled_sps:
            if 1 <= s <= W:
                mask_r[s - 1] = True
        a_r = float(a_abs[mask_r].sum()) if mask_r.any() else 0.0
        a_nr = float(a_abs[~mask_r].sum()) if (~mask_r).any() else 0.0
        return stopping_probability(a_r, a_nr, p.epsilon_d)


def sample_recalls(
    model: HierarchicalCMRModel,
    dataset_skeleton: Dataset,
    rng: np.random.Generator,
    *,
    recall_length_fn: Callable[[int], int] | None = None,
    max_recalls_per_list: int | None = None,
    state: EncodingState | None = None,
) -> pa.Table:
    """Deterministically sample a ``recalled.parquet``-shaped Table."""
    p = model.parameters
    if state is None:
        state = model.encode(dataset_skeleton)
    pdf = dataset_skeleton.presented.to_pandas()
    group_keys = sorted(set(zip(pdf["participant"].tolist(), pdf["list"].tolist())))

    rows: list[dict] = []
    for part, lst in group_keys:
        sub = pdf[(pdf["participant"] == part) & (pdf["list"] == lst)].sort_values(
            "serial_position"
        ).reset_index(drop=True)
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

        key = (int(part), int(lst))
        # After a recall, c_ret reactivates the recalled item's ENCODING
        # context (C&Z 2025 Fig 1b: "the retrieved item reactivates its
        # encoding context at retrieval"). This is the item-level context
        # c_item[picked_sp] at the time of encoding — NOT a drift from the
        # end-of-list cue. Using the encoding context as the starting
        # c_ret produces the canonical lag-CRP shape (forward asymmetry
        # concentrated near the recalled item, not smeared by end-of-list
        # recency).
        c_ret = state.c_item[key][picked_sp].copy()
        # Then drift by β_rec toward the input context e_{picked_sp} so that
        # successive recalls accumulate bias toward post-sp items.
        d_list = c_ret.shape[0]
        c_ret = drift_retrieval_context(
            c_ret, p.beta_rec, _onehot(d_list, picked_sp),
        )

        out_pos = 2
        while out_pos <= cap and len(recalled_sps) < W:
            if recall_length_fn is None and p.paradigm == "free_recall":
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
            # Reactivate c_ret to the encoding context of the newly-recalled
            # item (C&Z 2025 Fig 1b), then drift by β_rec toward the input.
            c_ret = state.c_item[key][picked_sp].copy()
            c_ret = drift_retrieval_context(
                c_ret, p.beta_rec, _onehot(c_ret.shape[0], picked_sp),
            )
            out_pos += 1

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
