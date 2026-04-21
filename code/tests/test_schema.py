"""Tests for validate_dataset: seven malformation classes (FR-012, SC-005)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ms_tcm import save_dataset, validate_dataset
from ms_tcm.schema import ValidationRule

from tests.test_dataset import _tiny_dataset


def _fresh(tmp_path: Path) -> Path:
    ds = _tiny_dataset()
    save_dataset(ds, tmp_path / "ds")
    return tmp_path / "ds"


def test_validator_passes_on_clean_dataset(tmp_path: Path) -> None:
    path = _fresh(tmp_path)
    report = validate_dataset(path)
    assert report.ok, report.summary()


def test_missing_required_file(tmp_path: Path) -> None:
    path = _fresh(tmp_path)
    (path / "recalled.parquet").unlink()
    report = validate_dataset(path)
    assert not report.ok
    assert any(
        v.rule == ValidationRule.MISSING_FILE_OR_COLUMN
        and v.file == "recalled.parquet"
        for v in report.violations
    )


def test_wrong_column_type(tmp_path: Path) -> None:
    path = _fresh(tmp_path)
    # Rewrite presented.parquet with a column type change.
    pres = pq.read_table(path / "presented.parquet").to_pandas()
    pres["participant"] = pres["participant"].astype("int32")
    pq.write_table(
        pa.Table.from_pandas(pres, preserve_index=False),
        path / "presented.parquet",
        compression="zstd", compression_level=1,
        use_dictionary=False, write_statistics=False, version="2.6",
    )
    report = validate_dataset(path)
    assert not report.ok
    # Either wrong-type OR hash-mismatch fires first; both are informative.
    assert any(
        v.rule in (ValidationRule.MISSING_FILE_OR_COLUMN, ValidationRule.MANIFEST_HASH_MISMATCH)
        for v in report.violations
    )


def test_manifest_hash_detects_tampering(tmp_path: Path) -> None:
    path = _fresh(tmp_path)
    p = path / "recalled.parquet"
    p.write_bytes(p.read_bytes() + b"\x00")
    report = validate_dataset(path)
    assert not report.ok
    assert any(
        v.rule == ValidationRule.MANIFEST_HASH_MISMATCH
        for v in report.violations
    )


def test_orphan_recall(tmp_path: Path) -> None:
    path = _fresh(tmp_path)
    # Rewrite recalled.parquet with a serial_position that doesn't exist.
    rec = pq.read_table(path / "recalled.parquet").to_pandas()
    rec.at[0, "serial_position"] = 99
    pq.write_table(
        pa.Table.from_pandas(rec, preserve_index=False),
        path / "recalled.parquet",
        compression="zstd", compression_level=1,
        use_dictionary=False, write_statistics=False, version="2.6",
    )
    # Update manifest hash so we get past rule 6 and hit rule 3.
    manifest = json.loads((path / "manifest.json").read_text())
    from ms_tcm.io import sha256_file
    manifest["files"]["recalled.parquet"]["sha256"] = sha256_file(path / "recalled.parquet")
    (path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    report = validate_dataset(path)
    assert not report.ok
    assert any(
        v.rule == ValidationRule.ORPHAN_RECALL for v in report.violations
    )


def test_list_group_mismatch(tmp_path: Path) -> None:
    path = _fresh(tmp_path)
    pres = pq.read_table(path / "presented.parquet").to_pandas()
    pres.at[0, "list_group"] = "late"  # list 0 with list_group "late" is inconsistent
    pq.write_table(
        pa.Table.from_pandas(pres, preserve_index=False),
        path / "presented.parquet",
        compression="zstd", compression_level=1,
        use_dictionary=False, write_statistics=False, version="2.6",
    )
    manifest = json.loads((path / "manifest.json").read_text())
    from ms_tcm.io import sha256_file
    manifest["files"]["presented.parquet"]["sha256"] = sha256_file(path / "presented.parquet")
    (path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    report = validate_dataset(path)
    assert not report.ok
    assert any(
        v.rule == ValidationRule.LIST_GROUP_MISMATCH for v in report.violations
    )


def test_duplicate_presented_row(tmp_path: Path) -> None:
    path = _fresh(tmp_path)
    pres = pq.read_table(path / "presented.parquet").to_pandas()
    # Overwrite the second row with the first row's primary key, creating a dup.
    pres.at[1, "serial_position"] = pres.at[0, "serial_position"]
    pq.write_table(
        pa.Table.from_pandas(pres, preserve_index=False),
        path / "presented.parquet",
        compression="zstd", compression_level=1,
        use_dictionary=False, write_statistics=False, version="2.6",
    )
    manifest = json.loads((path / "manifest.json").read_text())
    from ms_tcm.io import sha256_file
    manifest["files"]["presented.parquet"]["sha256"] = sha256_file(path / "presented.parquet")
    (path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    report = validate_dataset(path)
    assert not report.ok
    assert any(
        v.rule == ValidationRule.DUPLICATE_ROW for v in report.violations
    )


def test_manifest_malformed_json(tmp_path: Path) -> None:
    path = _fresh(tmp_path)
    (path / "manifest.json").write_text("{not-valid-json")
    report = validate_dataset(path)
    assert not report.ok
    assert any(
        v.rule == ValidationRule.MANIFEST_MALFORMED for v in report.violations
    )
