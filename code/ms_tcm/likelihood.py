"""Per-list and dataset-level log-likelihood.

See data-model.md section 4.3 and contracts/model-api.md section 8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from ms_tcm.dataset import Dataset
from ms_tcm.model import EncodingState, MSTCMModel
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
    """Return the log-likelihood of the observed recall sequence for one list.

    Extra-list intrusions (serial_position == 0) are skipped. An empty recall
    list contributes 0.
    """
    rec_df = recalled_list.to_pandas()
    if len(rec_df) == 0:
        return 0.0

    model = MSTCMModel(parameters)
    total = 0.0
    prev_sp: int | None = None
    for _, row in rec_df.sort_values("output_position").iterrows():
        sp = int(row["serial_position"])
        if sp == 0:
            continue  # extra-list intrusion; skip
        if prev_sp is None:
            probs = model.score_first_recall(encoding_state, participant, list_)
        else:
            probs = model.score_next_recall(
                encoding_state, participant, list_, prev_sp,
            )
        # sp is 1-based; presented candidates are 0..W-1 in serial-position order.
        idx = sp - 1
        if idx < 0 or idx >= probs.shape[0]:
            # Out-of-range serial position for this list; treat as infeasible.
            return float("-inf")
        p = float(probs[idx])
        if p <= 0.0 or not np.isfinite(p):
            return float("-inf")
        total += float(np.log(p))
        prev_sp = sp
    return total


def dataset_log_likelihood(
    dataset: Dataset,
    parameters: ModelParameters,
) -> float:
    """Sum of per-list log-likelihoods across every (participant, list)."""
    model = MSTCMModel(parameters)
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
    """Count recalls used vs intrusions excluded; useful for AIC/BIC and
    summary reporting."""
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
