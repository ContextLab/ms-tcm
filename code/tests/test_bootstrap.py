"""Bootstrap tests for v6 hierarchical CMR.

Covers: contracts/fitter.md §6 (parallel-equivalent determinism).

T035 / US2: verifying that running the multiprocessing-parallel path with
the same seed yields the same bootstrap_draws table as the serial path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from ms_tcm.bootstrap import bootstrap_ci
from ms_tcm.dataset import Dataset


def _tiny_dataset(n_participants: int = 3) -> Dataset:
    """Build a minimal multi-participant dataset with recalls so that the
    fit actually runs in a few seconds."""
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
            # 2-recall sequence per participant/list, forward order.
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


def test_parallel_matches_serial_same_seed() -> None:
    """FR-030 / contracts/fitter.md §6: the parallelized bootstrap must
    produce bit-identical per-draw MLE vectors as the serial path when the
    same seed is used. ``_run_one_bootstrap`` is deterministic given its
    work item, so ``imap_unordered`` + sort-by-bootstrap-id must match serial."""
    ds = _tiny_dataset(n_participants=3)

    # Serial (n_processes=1).
    r_serial = bootstrap_ci(
        ds, n_bootstraps=4, seed=1234, n_restarts=1,
        n_processes=1,
    )
    # Parallel (n_processes=2 when available).
    r_parallel = bootstrap_ci(
        ds, n_bootstraps=4, seed=1234, n_restarts=1,
        n_processes=2,
    )

    serial_df = r_serial.bootstrap_draws.to_pandas().sort_values("bootstrap_id")
    parallel_df = r_parallel.bootstrap_draws.to_pandas().sort_values("bootstrap_id")

    for col in serial_df.columns:
        if col in ("bootstrap_id", "converged"):
            assert (serial_df[col].to_numpy() == parallel_df[col].to_numpy()).all(), (
                f"parallel vs serial disagree on integer column {col!r}"
            )
        else:
            s = serial_df[col].to_numpy(dtype=np.float64)
            p = parallel_df[col].to_numpy(dtype=np.float64)
            np.testing.assert_allclose(
                s, p, atol=1e-12, rtol=0,
                err_msg=(
                    f"Parallel bootstrap diverges from serial on {col!r}: "
                    f"serial={s}, parallel={p}. "
                    f"Reference: contracts/fitter.md §6."
                ),
            )
