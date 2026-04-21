"""Tests for the bundled FRFR-category dataset and loader.

Covers US2 structural checks (T022) and US3 hash/provenance checks
(T031, T032, T032b, T033).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ms_tcm import load_dataset, load_frfr_category, validate_dataset
from ms_tcm.io import sha256_file
from ms_tcm.schema import REQUIRED_DATA_FILES

BUNDLED = Path(__file__).resolve().parents[2] / "data" / "raw" / "frfr_category"


@pytest.fixture
def bundled_manifest() -> dict:
    return json.loads((BUNDLED / "manifest.json").read_text())


def test_load_frfr_category_structure() -> None:
    ds = load_frfr_category()
    assert ds.num_participants == 30
    assert ds.num_lists_per_participant == 16
    assert ds.num_words_per_list == 16
    # Every list has exactly 4 unique categories.
    for p in range(ds.num_participants):
        for lst in range(ds.num_lists_per_participant):
            assert len(ds.categories_in_list(p, lst)) == 4, (
                f"participant={p}, list={lst} has != 4 categories"
            )


def test_bundled_dataset_row_counts(bundled_manifest: dict) -> None:
    rc = bundled_manifest["row_counts"]
    assert rc["presented"] == 7680  # 30 * 16 * 16
    # Intrusion + in-list counts sum to the total.
    assert rc["recalled_in_list"] + rc["recalled_extra_list_intrusions"] == rc["recalled_total"]


def test_shipped_dataset_hashes_match_manifest(bundled_manifest: dict) -> None:
    """Primary guard against accidental data drift (FR-011, SC-003)."""
    for fname, entry in bundled_manifest["files"].items():
        expected = entry["sha256"]
        actual = sha256_file(BUNDLED / fname)
        assert actual == expected, (
            f"{fname} sha256 drift: manifest={expected}, disk={actual}"
        )


def test_shipped_dataset_has_all_required_files(bundled_manifest: dict) -> None:
    """T032b: dataset directory contains exactly the files in the manifest."""
    listed = set(bundled_manifest["files"].keys())
    for name in REQUIRED_DATA_FILES:
        assert name in listed, f"required file {name} missing from manifest"
    # Plus manifest.json itself.
    disk = {p.name for p in BUNDLED.iterdir() if p.is_file()}
    # All listed files must be on disk; extras like exp2.egg.cache are allowed
    # (gitignored) but not required.
    missing_on_disk = listed - disk
    assert not missing_on_disk, f"files in manifest but not on disk: {missing_on_disk}"


def test_bundled_dataset_validates_clean() -> None:
    """The shipped dataset passes all seven validation rules."""
    report = validate_dataset(BUNDLED)
    assert report.ok, report.summary()


def test_manifest_hashes_match_file_contents_after_roundtrip(tmp_path: Path) -> None:
    """T033: a save-then-load round-trip preserves manifest-hash agreement."""
    from ms_tcm import save_dataset
    ds = load_frfr_category()
    save_dataset(ds, tmp_path / "copy")
    # Validate the rewritten copy.
    report = validate_dataset(tmp_path / "copy")
    assert report.ok, report.summary()
