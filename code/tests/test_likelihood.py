"""Unit tests for the per-list and dataset log-likelihood (FR-017)."""

from __future__ import annotations

import numpy as np

from ms_tcm import ModelParameters
from ms_tcm.likelihood import list_log_likelihood, dataset_log_likelihood

from tests.test_dataset import _tiny_dataset


def test_empty_recall_contributes_zero_log_likelihood() -> None:
    """A list with no recalls contributes 0 to the log-likelihood."""
    ds = _tiny_dataset()
    # Empty the recall table.
    import pyarrow as pa, pandas as pd
    empty = pa.Table.from_pandas(
        pd.DataFrame({c: pd.Series(dtype=d) for c, d in [
            ("participant", "int64"), ("list", "int64"),
            ("output_position", "int64"), ("word", "string"),
            ("category", "string"), ("serial_position", "int64"),
            ("list_group", "string"),
        ]}),
        preserve_index=False,
    )
    from ms_tcm import Dataset
    ds_no_recall = Dataset(presented=ds.presented, recalled=empty, manifest=ds.manifest)
    ll = dataset_log_likelihood(ds_no_recall, ModelParameters())
    assert ll == 0.0


def test_intrusions_skipped_in_likelihood() -> None:
    """Intrusions (serial_position == 0) do not contribute to log-likelihood."""
    import pyarrow as pa, pandas as pd
    from ms_tcm import Dataset
    ds = _tiny_dataset()
    rec_df = ds.recalled.to_pandas().copy()
    rec_df = pd.concat([rec_df, pd.DataFrame([{
        "participant": 0, "list": 0, "output_position": 2,
        "word": "XXXXX", "category": "",
        "serial_position": 0, "list_group": "early",
    }])], ignore_index=True)
    ds_intr = Dataset(
        presented=ds.presented,
        recalled=pa.Table.from_pandas(rec_df, preserve_index=False),
        manifest=ds.manifest,
    )
    ll_with_intr = dataset_log_likelihood(ds_intr, ModelParameters())
    ll_no_intr = dataset_log_likelihood(ds, ModelParameters())
    assert abs(ll_with_intr - ll_no_intr) < 1e-12


def test_log_likelihood_is_finite_on_tiny_dataset() -> None:
    ds = _tiny_dataset()
    ll = dataset_log_likelihood(ds, ModelParameters())
    assert np.isfinite(ll)
    # Probability of a specific recall is < 1, so log-likelihood is < 0.
    assert ll <= 0.0
