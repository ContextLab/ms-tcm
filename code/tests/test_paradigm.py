"""Paradigm-selection test (T019b / US1 / FR-006).

Under the cued-recall paradigm (ModelParameters(paradigm="cued_recall")),
β_rec retrieval drift and ε_d stopping rule MUST NOT affect the scored
probability distribution, per v6 §5.1. These parameters govern within-trial
dynamics that have no causal role in single-response cued recall.

The test constructs two cued-recall models with identical β_enc, β_story,
γ_fc, k, λ but radically different β_rec and ε_d values, and verifies
``score_cue`` returns bit-identical probabilities.

Reference: notes/two_level_cmr_v6.pdf §5.1; Cornell et al. 2024 cueing
mechanism (v6 §7.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from ms_tcm.dataset import Dataset
from ms_tcm.hcmr import HierarchicalCMRModel
from ms_tcm.params import ModelParameters


def _build_cued_dataset() -> Dataset:
    """A 1-participant 1-list 4-word single-storyline cued-recall dataset."""
    presented = pd.DataFrame(
        [
            {
                "participant": 0, "list": 0, "serial_position": sp,
                "word": f"W{sp}", "category": "BODY PARTS", "size": "small",
                "first_letter": chr(ord("A") + sp - 1), "word_length": 6,
                "color_r": 0, "color_g": 0, "color_b": 0,
                "pos_x": 0.0, "pos_y": 0.0,
            }
            for sp in range(1, 5)
        ]
    )
    recalled = pd.DataFrame(
        columns=[
            "participant", "list", "output_position", "word", "category",
            "serial_position", "list_group",
        ]
    )
    return Dataset(
        presented=pa.Table.from_pandas(presented, preserve_index=False),
        recalled=pa.Table.from_pandas(recalled, preserve_index=False),
        manifest={"schema_version": "1.0.0", "dataset_name": "_test_cued"},
    )


def test_cued_recall_score_cue_ignores_beta_rec() -> None:
    """Changing β_rec in the cued-recall paradigm MUST NOT change
    ``score_cue`` output, since β_rec governs between-recall drift that
    doesn't fire in single-response retrieval (v6 §5.1)."""
    common = dict(
        beta_enc=0.679, beta_story=0.4, gamma_fc=0.315, k=6.5,
        lambda_reinstate=0.8, paradigm="cued_recall", feature_dim=71,
    )
    # In cued-recall paradigm, validation relaxes beta_rec/epsilon_d
    # constraints; we still set sane defaults so the dataclass accepts them.
    params_a = ModelParameters(**common, beta_rec=0.1, epsilon_d=0.5)
    params_b = ModelParameters(**common, beta_rec=0.9, epsilon_d=5.0)

    ds = _build_cued_dataset()
    model_a = HierarchicalCMRModel(params_a)
    model_b = HierarchicalCMRModel(params_b)
    state_a = model_a.encode(ds)
    state_b = model_b.encode(ds)

    for cue_sp in range(1, 5):
        probs_a = model_a.score_cue(state_a, participant=0, list_=0, cue_serial_position=cue_sp)
        probs_b = model_b.score_cue(state_b, participant=0, list_=0, cue_serial_position=cue_sp)
        np.testing.assert_allclose(
            probs_a, probs_b, atol=1e-12,
            err_msg=(
                f"score_cue changed under different beta_rec / epsilon_d "
                f"(v6 §5.1 violation) at cue_sp={cue_sp}"
            ),
        )


def test_cued_recall_score_cue_returns_valid_distribution() -> None:
    """``score_cue`` returns a non-negative probability distribution that
    sums to 1 (v6 §3.1 Eq 9)."""
    params = ModelParameters(paradigm="cued_recall")
    ds = _build_cued_dataset()
    model = HierarchicalCMRModel(params)
    state = model.encode(ds)

    for cue_sp in range(1, 5):
        probs = model.score_cue(state, 0, 0, cue_sp)
        assert np.all(probs >= 0), "probabilities must be non-negative"
        assert abs(float(probs.sum()) - 1.0) < 1e-10, \
            "probabilities must sum to 1"


def test_cued_recall_score_cue_rejects_out_of_range() -> None:
    """``score_cue`` validates cue_serial_position is in [1, W]."""
    params = ModelParameters(paradigm="cued_recall")
    ds = _build_cued_dataset()
    model = HierarchicalCMRModel(params)
    state = model.encode(ds)

    import pytest as _pytest
    with _pytest.raises(ValueError):
        model.score_cue(state, 0, 0, 0)
    with _pytest.raises(ValueError):
        model.score_cue(state, 0, 0, 100)
