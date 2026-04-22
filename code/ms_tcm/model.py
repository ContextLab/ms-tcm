"""MSTCMModel orchestrator + EncodingState + sample_recalls.

See contracts/model-api.md sections 4, 10 and data-model.md sections 4.1-4.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa

from ms_tcm.composite import compose_encoding, compose_retrieval
from ms_tcm.context import update_global_context, update_storyline_context
from ms_tcm.dataset import Dataset
from ms_tcm.features import encode_features, list_start_vector
from ms_tcm.mechanisms import apply_resumption_reinstatement, interference_factor
from ms_tcm.params import ModelParameters
from ms_tcm.similarity import cosine_similarity, recall_probabilities


@dataclass(frozen=True)
class EncodingState:
    # Each dict is keyed by (participant, list_).
    c_global: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    c_storyline: dict[tuple[int, int], dict[str, np.ndarray]] = field(default_factory=dict)
    c_composite: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    active_storyline: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    list_presented: dict[tuple[int, int], pd.DataFrame] = field(default_factory=dict)
    list_features: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)


class MSTCMModel:
    """Forward simulator for MS-TCM.

    See data-model.md section 4.1 for the encoding state machine and section 4.2
    for the retrieval scoring used by score_first_recall / score_next_recall.
    """

    def __init__(self, parameters: ModelParameters) -> None:
        self.parameters = parameters

    def encode(self, dataset: Dataset) -> EncodingState:
        p = self.parameters
        state = EncodingState()

        pdf = dataset.presented.to_pandas()
        feat_matrix = encode_features(pdf)
        d = feat_matrix.shape[1]
        if p.feature_dim != d:
            raise ValueError(
                f"ModelParameters.feature_dim={p.feature_dim} does not match "
                f"dataset's encoded feature dimensionality {d}; "
                "the dataset's feature encoder width and the parameter set "
                "must agree (see data-model.md section 2)."
            )

        groups = sorted(
            set(zip(pdf["participant"].tolist(), pdf["list"].tolist()))
        )
        storyline_mode = p.w_storyline > 0.0

        for part, lst in groups:
            mask = (pdf["participant"] == part) & (pdf["list"] == lst)
            sub = pdf.loc[mask].sort_values("serial_position").reset_index(drop=False)
            W = len(sub)
            d = feat_matrix.shape[1]

            e_start = list_start_vector()
            c_g_traj = np.zeros((W + 1, d), dtype=np.float64)
            c_g_traj[0] = e_start

            categories = sorted(set(sub["category"].tolist()))
            c_s_traj: dict[str, np.ndarray] = {}
            for cat in categories:
                arr = np.zeros((W + 1, d), dtype=np.float64)
                arr[0] = e_start
                c_s_traj[cat] = arr

            c_comp = np.zeros((W, d), dtype=np.float64)
            active = np.empty(W, dtype=object)

            for t in range(W):
                row = sub.iloc[t]
                orig_idx = int(row["index"])
                c_in = feat_matrix[orig_idx]
                cat = row["category"]

                c_g_new = update_global_context(c_g_traj[t], p.beta_global, c_in)
                c_g_traj[t + 1] = c_g_new

                for other_cat in categories:
                    if other_cat == cat and storyline_mode:
                        updated = update_storyline_context(
                            c_s_traj[other_cat][t], p.beta_storyline, c_in,
                        )
                        if p.gamma > 0.0:
                            updated = updated + apply_resumption_reinstatement(
                                c_s_traj[other_cat][t], p.gamma,
                            )
                        c_s_traj[other_cat][t + 1] = updated
                    else:
                        c_s_traj[other_cat][t + 1] = c_s_traj[other_cat][t]

                if storyline_mode:
                    c_comp[t] = compose_encoding(
                        c_g_traj[t + 1], c_s_traj[cat][t + 1],
                        p.w_global, p.w_storyline,
                    )
                else:
                    c_comp[t] = c_g_traj[t + 1].copy()
                active[t] = cat

            key = (int(part), int(lst))
            state.c_global[key] = c_g_traj
            state.c_storyline[key] = c_s_traj
            state.c_composite[key] = c_comp
            state.active_storyline[key] = active
            state.list_presented[key] = sub.drop(columns=["index"]).reset_index(drop=True)
            list_feats = np.stack([feat_matrix[int(i)] for i in sub["index"].tolist()])
            state.list_features[key] = list_feats

        return state

    def _retrieval_context_end_of_list(
        self, state: EncodingState, participant: int, list_: int,
    ) -> np.ndarray:
        p = self.parameters
        key = (participant, list_)
        c_g_end = state.c_global[key][-1]
        if p.w_storyline > 0.0:
            active = state.active_storyline[key]
            c_s_traj = state.c_storyline[key]
            last_cat = active[-1]
            c_s_end = c_s_traj[last_cat][-1]
            return compose_retrieval(
                c_g_end, c_s_end, p.w_global_ret, p.w_storyline_ret,
            )
        return c_g_end.copy()

    def _retrieval_context_after(
        self, state: EncodingState, participant: int, list_: int,
        last_recalled_serial_position: int,
    ) -> np.ndarray:
        p = self.parameters
        key = (participant, list_)
        t = last_recalled_serial_position - 1
        c_g_t = state.c_global[key][t + 1]
        if p.w_storyline > 0.0:
            presented = state.list_presented[key]
            cat = str(presented.iloc[t]["category"])
            c_s_t = state.c_storyline[key][cat][t + 1]
            return compose_retrieval(
                c_g_t, c_s_t, p.w_global_ret, p.w_storyline_ret,
            )
        return c_g_t.copy()

    def _score_against_candidates(
        self, c_ret: np.ndarray, state: EncodingState,
        participant: int, list_: int,
        cue_serial_position: int,
    ) -> np.ndarray:
        p = self.parameters
        key = (participant, list_)
        candidates = state.c_composite[key]    # shape (W, d)

        # Vectorized cosine: dot product row-wise divided by ||c_ret|| * ||row||.
        # Matches cosine_similarity()'s epsilon-guarded definition: zero-norm
        # rows yield 0.0 rather than NaN.
        eps = 1e-30
        c_ret_norm = float(np.linalg.norm(c_ret))
        cand_norms = np.linalg.norm(candidates, axis=1)
        if c_ret_norm == 0.0:
            sims = np.zeros(candidates.shape[0], dtype=np.float64)
        else:
            safe_cand_norms = np.where(cand_norms == 0.0, 1.0, cand_norms + eps)
            sims = candidates @ c_ret / ((c_ret_norm + eps) * safe_cand_norms)
            sims = np.where(cand_norms == 0.0, 0.0, sims)

        if p.lambda_interference != 0.0:
            # §5.3: unnormalized score = sim · exp(-λ · I_{ij}).
            # For free recall we instantiate I_{ij} as the count of items
            # encoded between candidate i (at serial position s_i) and the
            # cue event j (at serial position cue_serial_position): i.e.
            # max(0, |s_i - s_j| - 1). This is the release-from-proactive-
            # interference interpretation in notes/ms-tcm.pdf §5.3 — more
            # intervening items ⇒ more interference ⇒ smaller factor.
            W = candidates.shape[0]
            serial_positions = np.arange(1, W + 1, dtype=np.float64)
            gap = np.abs(serial_positions - float(cue_serial_position))
            i_ij = np.maximum(gap - 1.0, 0.0)
            factor = np.exp(-float(p.lambda_interference) * i_ij)
            sims = sims * factor
        # Primacy gradient (Polyn et al. 2009, Eq. 8): item-i activation bias
        # phi_i = phi_s * exp(-phi_d * (i-1)) + 1 boosts early-list items,
        # producing the primacy limb of the serial-position curve. Vanilla
        # TCM has no primacy mechanism; without this the model only
        # predicts recency. phi_s = 0 disables primacy (default).
        if p.phi_s > 0.0:
            W = sims.shape[0]
            positions = np.arange(W, dtype=np.float64)  # 0-indexed so i=1 -> 0
            primacy = float(p.phi_s) * np.exp(-float(p.phi_d) * positions) + 1.0
            sims = sims * primacy
        return recall_probabilities(sims, tau=p.tau)

    def score_first_recall(
        self, state: EncodingState, participant: int, list_: int,
    ) -> np.ndarray:
        key = (participant, list_)
        W = state.c_composite[key].shape[0]
        c_ret = self._retrieval_context_end_of_list(state, participant, list_)
        # First recall is cued by the end-of-list context; take j = W (last
        # encoded position) so I_{ij} measures distance back into the list.
        return self._score_against_candidates(
            c_ret, state, participant, list_, cue_serial_position=W,
        )

    def score_next_recall(
        self, state: EncodingState, participant: int, list_: int,
        last_recalled_serial_position: int,
    ) -> np.ndarray:
        if last_recalled_serial_position < 1:
            raise ValueError(
                f"last_recalled_serial_position must be 1-based positive; "
                f"got {last_recalled_serial_position!r}"
            )
        c_ret = self._retrieval_context_after(
            state, participant, list_, last_recalled_serial_position,
        )
        return self._score_against_candidates(
            c_ret, state, participant, list_,
            cue_serial_position=last_recalled_serial_position,
        )


def sample_recalls(
    model: MSTCMModel,
    dataset_skeleton: Dataset,
    rng: np.random.Generator,
    *,
    recall_length_fn: Callable[[int], int] | None = None,
) -> pa.Table:
    """Sample a synthetic recalled.parquet-shaped table from MS-TCM probabilities."""
    state = model.encode(dataset_skeleton)
    pdf = dataset_skeleton.presented.to_pandas()
    groups = sorted(set(zip(pdf["participant"].tolist(), pdf["list"].tolist())))

    rows: list[dict] = []
    for part, lst in groups:
        mask = (pdf["participant"] == part) & (pdf["list"] == lst)
        sub = pdf.loc[mask].sort_values("serial_position").reset_index(drop=True)
        W = len(sub)
        R = W if recall_length_fn is None else int(recall_length_fn(W))
        R = max(0, min(R, W))

        if R == 0:
            continue

        first_probs = model.score_first_recall(state, int(part), int(lst))
        pick = int(rng.choice(W, p=first_probs))
        picked_sp = int(sub.iloc[pick]["serial_position"])
        list_group = "early" if int(lst) < 8 else "late"
        rows.append({
            "participant": int(part),
            "list": int(lst),
            "output_position": 1,
            "word": str(sub.iloc[pick]["word"]),
            "category": str(sub.iloc[pick]["category"]),
            "serial_position": picked_sp,
            "list_group": list_group,
        })

        for out_pos in range(2, R + 1):
            next_probs = model.score_next_recall(
                state, int(part), int(lst), picked_sp,
            )
            pick = int(rng.choice(W, p=next_probs))
            picked_sp = int(sub.iloc[pick]["serial_position"])
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
