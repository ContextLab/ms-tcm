"""Cross-platform numerical tolerance tests for the Tier 1 fitter.

Spec: specs/002-ms-tcm-v6-hcmr/spec.md FR-023 / SC-006.
Contract: specs/002-ms-tcm-v6-hcmr/contracts/fitter.md §9.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from ms_tcm.dataset import Dataset
from ms_tcm.fit import fit_mle


def _tiny_dataset(n_participants: int = 3) -> Dataset:
    rows_pres: list[dict] = []
    rows_rec: list[dict] = []
    W = 4
    for p in range(n_participants):
        for L in range(1):
            for sp in range(1, W + 1):
                rows_pres.append({
                    "participant": p, "list": L, "serial_position": sp,
                    "word": f"W{sp}", "category": "BODY PARTS", "size": "small",
                    "first_letter": chr(ord("A") + (sp - 1)), "word_length": 5,
                    "color_r": 0, "color_g": 0, "color_b": 0,
                    "pos_x": 0.0, "pos_y": 0.0, "list_group": "early",
                })
            for out_pos, sp in enumerate([W, W - 1], start=1):
                rows_rec.append({
                    "participant": p, "list": L, "output_position": out_pos,
                    "word": f"W{sp}", "category": "BODY PARTS",
                    "serial_position": sp, "list_group": "early",
                })
    pres = pa.Table.from_pandas(pd.DataFrame(rows_pres), preserve_index=False)
    rec = pa.Table.from_pandas(pd.DataFrame(rows_rec), preserve_index=False)
    return Dataset(
        presented=pres, recalled=rec,
        manifest={"schema_version": "1.0.0", "dataset_name": "_tiny"},
    )


def test_tier1_mle_within_1e_minus_10() -> None:
    """FR-023: Tier 1 fit run twice with the same seed must return MLE
    vectors within 1e-10 absolute tolerance (determinism anchor). This is
    the cross-platform tolerance bar — the same code + seed + data must
    produce identical floats on macOS, Ubuntu, and Windows to within 1e-10.
    """
    ds = _tiny_dataset(n_participants=3)
    fit_a = fit_mle(ds, n_restarts=2, seed=7)
    fit_b = fit_mle(ds, n_restarts=2, seed=7)

    for name in fit_a.parameters:
        a = float(fit_a.parameters[name]["mle"])
        b = float(fit_b.parameters[name]["mle"])
        assert abs(a - b) < 1e-10, (
            f"Non-deterministic MLE on {name}: fit A={a!r}, fit B={b!r}. "
            f"|Δ|={abs(a - b):.3e} exceeds 1e-10 tolerance (FR-023)."
        )
