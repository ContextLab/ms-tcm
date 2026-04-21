"""Tests for Dataset save/load round-trip (FR-008)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from ms_tcm import Dataset, load_dataset, save_dataset


def _tiny_dataset() -> Dataset:
    pres = pa.Table.from_pandas(pd.DataFrame([
        {"participant": 0, "list": 0, "serial_position": 1,
         "word": "APPLE", "category": "FRUITS", "size": "small",
         "first_letter": "A", "word_length": 5,
         "color_r": -1, "color_g": -1, "color_b": -1,
         "pos_x": float("nan"), "pos_y": float("nan"),
         "list_group": "early"},
        {"participant": 0, "list": 0, "serial_position": 2,
         "word": "MANGO", "category": "FRUITS", "size": "small",
         "first_letter": "M", "word_length": 5,
         "color_r": -1, "color_g": -1, "color_b": -1,
         "pos_x": float("nan"), "pos_y": float("nan"),
         "list_group": "early"},
    ]), preserve_index=False)
    rec = pa.Table.from_pandas(pd.DataFrame([
        {"participant": 0, "list": 0, "output_position": 1,
         "word": "MANGO", "category": "FRUITS",
         "serial_position": 2, "list_group": "early"},
    ]), preserve_index=False)
    manifest = {
        "source": {"paper": "t", "condition": "t", "egg_url": "t", "egg_sha256": "t"},
        "design": {
            "participants": 1, "lists_per_participant": 1, "words_per_list": 2,
            "unique_categories_per_list": 1, "unique_categories_total": 1,
            "early_lists": "list<8", "late_lists": "list>=8",
        },
        "files": {},
        "row_counts": {
            "presented": 2, "recalled_total": 1,
            "recalled_in_list": 1, "recalled_extra_list_intrusions": 0,
        },
        "created_at": "2026-04-21T00:00:00+00:00",
    }
    return Dataset(presented=pres, recalled=rec, manifest=manifest)


def test_roundtrip_save_load_is_lossless(tmp_path: Path) -> None:
    ds = _tiny_dataset()
    save_dataset(ds, tmp_path / "ds")
    loaded = load_dataset(tmp_path / "ds")
    pd.testing.assert_frame_equal(
        ds.presented.to_pandas(), loaded.presented.to_pandas(),
    )
    pd.testing.assert_frame_equal(
        ds.recalled.to_pandas(), loaded.recalled.to_pandas(),
    )
    # Manifest files block is populated on save; other blocks are preserved.
    for key in ("source", "design", "row_counts"):
        assert ds.manifest[key] == loaded.manifest[key]


def test_load_detects_tampered_file(tmp_path: Path) -> None:
    ds = _tiny_dataset()
    save_dataset(ds, tmp_path / "ds")
    # Mutate the presented parquet file after it was written.
    p = tmp_path / "ds" / "presented.parquet"
    data = p.read_bytes()
    p.write_bytes(data + b"\x00")
    import pytest
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        load_dataset(tmp_path / "ds")
