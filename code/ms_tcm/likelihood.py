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
    for _, row in rec_df.sort_values("output_position").iterrows():
        sp = int(row["serial_position"])
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
    """Sum of list_log_likelihood across every (participant, list)."""
    model = HierarchicalCMRModel(parameters)
    state = model.encode(dataset)
    total = 0.0
    for part, lst, pres_sub, rec_sub in dataset.iter_lists():
        contrib = list_log_likelihood(
            pres_sub, rec_sub, parameters, state, part, lst,
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
