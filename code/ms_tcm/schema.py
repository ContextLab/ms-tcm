"""Dataset schemas, manifest contract, and validation routine.

See:
- ``specs/001-ms-tcm-impl/contracts/dataset-schema.md`` §3–§5
- ``specs/001-ms-tcm-impl/data-model.md`` §7

This module owns the single source of truth for the on-disk layout
(Constitution Principle II).
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa

from ms_tcm.io import read_json, sha256_file


# ---- Arrow schemas (contracts/dataset-schema.md §3) ---------------------------

PRESENTED_SCHEMA: pa.Schema = pa.schema([
    ("participant", pa.int64()),
    ("list", pa.int64()),
    ("serial_position", pa.int64()),
    ("word", pa.string()),
    ("category", pa.string()),
    ("size", pa.string()),
    ("first_letter", pa.string()),
    ("word_length", pa.int64()),
    ("color_r", pa.int64()),
    ("color_g", pa.int64()),
    ("color_b", pa.int64()),
    ("pos_x", pa.float64()),
    ("pos_y", pa.float64()),
    ("list_group", pa.string()),
])

RECALLED_SCHEMA: pa.Schema = pa.schema([
    ("participant", pa.int64()),
    ("list", pa.int64()),
    ("output_position", pa.int64()),
    ("word", pa.string()),
    ("category", pa.string()),
    ("serial_position", pa.int64()),
    ("list_group", pa.string()),
])


# ---- Manifest key contract (FR-009; resolves C2) -----------------------------

MANIFEST_TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "source", "design", "files", "row_counts", "created_at",
})

MANIFEST_SOURCE_KEYS: frozenset[str] = frozenset({
    "paper", "condition", "egg_url", "egg_sha256",
})

MANIFEST_DESIGN_KEYS: frozenset[str] = frozenset({
    "participants",
    "lists_per_participant",
    "words_per_list",
    "unique_categories_per_list",
    "unique_categories_total",
    "early_lists",
    "late_lists",
})

MANIFEST_ROW_COUNT_KEYS: frozenset[str] = frozenset({
    "presented",
    "recalled_total",
    "recalled_in_list",
    "recalled_extra_list_intrusions",
})

MANIFEST_FILE_ENTRY_KEYS: frozenset[str] = frozenset({"sha256", "size_bytes"})

def _types_compatible(actual: pa.DataType, expected: pa.DataType) -> bool:
    """string and large_string are pandas-interchange equivalents; treat as equal."""
    if actual == expected:
        return True
    string_types = (pa.string(), pa.large_string())
    if actual in string_types and expected in string_types:
        return True
    return False


REQUIRED_DATA_FILES: tuple[str, ...] = (
    "presented.parquet",
    "recalled.parquet",
    "presented.csv",
    "recalled.csv",
)


# ---- Validation rules (FR-012; data-model.md §7) -----------------------------

class ValidationRule(enum.Enum):
    """Seven malformation classes the validator must catch."""

    MISSING_FILE_OR_COLUMN = 1
    PRESENTED_ROW_INCONSISTENT_WITH_DESIGN = 2
    ORPHAN_RECALL = 3
    LIST_GROUP_MISMATCH = 4
    DUPLICATE_ROW = 5
    MANIFEST_HASH_MISMATCH = 6
    MANIFEST_MALFORMED = 7


@dataclass(frozen=True)
class Violation:
    rule: ValidationRule
    message: str
    file: str | None = None
    column: str | None = None
    row: int | None = None


@dataclass
class ValidationReport:
    ok: bool
    violations: list[Violation] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return "OK"
        lines = [f"{len(self.violations)} violation(s):"]
        for v in self.violations:
            loc = []
            if v.file:
                loc.append(f"file={v.file}")
            if v.column:
                loc.append(f"col={v.column}")
            if v.row is not None:
                loc.append(f"row={v.row}")
            locstr = " (" + ", ".join(loc) + ")" if loc else ""
            lines.append(f"  [{v.rule.name}]{locstr} {v.message}")
        return "\n".join(lines)


def _check_manifest_shape(manifest: Any) -> list[Violation]:
    """Rule 7: manifest is well-formed with the required top-level + sub keys."""
    v: list[Violation] = []
    if not isinstance(manifest, dict):
        v.append(Violation(
            ValidationRule.MANIFEST_MALFORMED,
            "manifest is not a JSON object",
            file="manifest.json",
        ))
        return v
    missing = MANIFEST_TOP_LEVEL_KEYS - manifest.keys()
    extra = manifest.keys() - MANIFEST_TOP_LEVEL_KEYS
    if missing:
        v.append(Violation(
            ValidationRule.MANIFEST_MALFORMED,
            f"manifest missing required top-level keys: {sorted(missing)}",
            file="manifest.json",
        ))
    if extra:
        v.append(Violation(
            ValidationRule.MANIFEST_MALFORMED,
            f"manifest has unexpected top-level keys: {sorted(extra)}",
            file="manifest.json",
        ))
    # Drill into sub-objects only when the top-level keys exist
    for top, sub_keys in (
        ("source", MANIFEST_SOURCE_KEYS),
        ("design", MANIFEST_DESIGN_KEYS),
        ("row_counts", MANIFEST_ROW_COUNT_KEYS),
    ):
        if top in manifest and isinstance(manifest[top], dict):
            smissing = sub_keys - manifest[top].keys()
            if smissing:
                v.append(Violation(
                    ValidationRule.MANIFEST_MALFORMED,
                    f"manifest.{top} missing keys: {sorted(smissing)}",
                    file="manifest.json",
                ))
    if "files" in manifest and isinstance(manifest["files"], dict):
        for fname, entry in manifest["files"].items():
            if not isinstance(entry, dict) or not MANIFEST_FILE_ENTRY_KEYS.issubset(entry.keys()):
                v.append(Violation(
                    ValidationRule.MANIFEST_MALFORMED,
                    f"manifest.files[{fname!r}] missing sha256/size_bytes",
                    file="manifest.json",
                ))
    return v


def _check_files_and_schemas(dataset_dir: Path, manifest: dict) -> list[Violation]:
    """Rule 1: every required file present with the expected schema."""
    v: list[Violation] = []
    for fname in REQUIRED_DATA_FILES:
        if not (dataset_dir / fname).exists():
            v.append(Violation(
                ValidationRule.MISSING_FILE_OR_COLUMN,
                f"required file missing: {fname}",
                file=fname,
            ))
    if not (dataset_dir / "manifest.json").exists():
        v.append(Violation(
            ValidationRule.MISSING_FILE_OR_COLUMN,
            "required file missing: manifest.json",
            file="manifest.json",
        ))
    # Parquet schema checks
    for fname, expected in (
        ("presented.parquet", PRESENTED_SCHEMA),
        ("recalled.parquet", RECALLED_SCHEMA),
    ):
        path = dataset_dir / fname
        if not path.exists():
            continue
        try:
            import pyarrow.parquet as pq
            tbl = pq.read_table(path)
        except Exception as exc:
            v.append(Violation(
                ValidationRule.MISSING_FILE_OR_COLUMN,
                f"could not read parquet: {exc!r}",
                file=fname,
            ))
            continue
        # Check that every expected column is present with the expected type.
        schema = tbl.schema
        for f in expected:
            idx = schema.get_field_index(f.name)
            if idx < 0:
                v.append(Violation(
                    ValidationRule.MISSING_FILE_OR_COLUMN,
                    f"column {f.name!r} missing in {fname}",
                    file=fname, column=f.name,
                ))
                continue
            actual_type = schema.field(idx).type
            if not _types_compatible(actual_type, f.type):
                v.append(Violation(
                    ValidationRule.MISSING_FILE_OR_COLUMN,
                    f"column {f.name!r} has type {actual_type}, expected {f.type}",
                    file=fname, column=f.name,
                ))
    return v


def _check_hashes(dataset_dir: Path, manifest: dict) -> list[Violation]:
    """Rule 6: every file's SHA-256 matches the manifest."""
    v: list[Violation] = []
    if "files" not in manifest or not isinstance(manifest["files"], dict):
        return v
    for fname, entry in manifest["files"].items():
        if not isinstance(entry, dict) or "sha256" not in entry:
            continue
        path = dataset_dir / fname
        if not path.exists():
            v.append(Violation(
                ValidationRule.MISSING_FILE_OR_COLUMN,
                f"manifest lists {fname} but file is missing",
                file=fname,
            ))
            continue
        actual = sha256_file(path)
        expected = entry["sha256"]
        if actual != expected:
            v.append(Violation(
                ValidationRule.MANIFEST_HASH_MISMATCH,
                f"{fname} sha256 mismatch: manifest={expected}, disk={actual}",
                file=fname,
            ))
    return v


def _check_design_consistency(dataset_dir: Path, manifest: dict) -> list[Violation]:
    """Rule 2: presented rows consistent with manifest.design.

    Rule 4: list_group matches list<8 vs list>=8.
    Rule 5: no duplicate primary keys.
    """
    v: list[Violation] = []
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return v
    path = dataset_dir / "presented.parquet"
    if not path.exists():
        return v
    try:
        df = pq.read_table(path).to_pandas()
    except Exception:
        return v

    design = manifest.get("design", {})
    n_pts_expected = design.get("participants")
    lists_per_pt = design.get("lists_per_participant")
    words_per_list = design.get("words_per_list")

    if n_pts_expected is not None:
        n_pts_actual = int(df["participant"].nunique())
        if n_pts_actual != n_pts_expected:
            v.append(Violation(
                ValidationRule.PRESENTED_ROW_INCONSISTENT_WITH_DESIGN,
                f"participants: manifest={n_pts_expected}, data={n_pts_actual}",
                file="presented.parquet", column="participant",
            ))
    if lists_per_pt is not None:
        bad = df.groupby("participant")["list"].nunique()
        bad = bad[bad != lists_per_pt]
        if len(bad) > 0:
            v.append(Violation(
                ValidationRule.PRESENTED_ROW_INCONSISTENT_WITH_DESIGN,
                f"lists_per_participant mismatch for {len(bad)} participant(s)",
                file="presented.parquet", column="list",
            ))
    if words_per_list is not None:
        sizes = df.groupby(["participant", "list"]).size()
        bad = sizes[sizes != words_per_list]
        if len(bad) > 0:
            v.append(Violation(
                ValidationRule.PRESENTED_ROW_INCONSISTENT_WITH_DESIGN,
                f"words_per_list mismatch for {len(bad)} (participant, list) group(s)",
                file="presented.parquet", column="serial_position",
            ))

    # Rule 4: list_group must match list<8 vs list>=8
    lg_bad = df[~(
        ((df["list"] < 8) & (df["list_group"] == "early"))
        | ((df["list"] >= 8) & (df["list_group"] == "late"))
    )]
    if len(lg_bad) > 0:
        v.append(Violation(
            ValidationRule.LIST_GROUP_MISMATCH,
            f"{len(lg_bad)} presented row(s) have list_group inconsistent with list index",
            file="presented.parquet", column="list_group",
        ))

    # Rule 5: no duplicates
    dup = df.duplicated(subset=["participant", "list", "serial_position"])
    if dup.any():
        v.append(Violation(
            ValidationRule.DUPLICATE_ROW,
            f"{int(dup.sum())} duplicate (participant, list, serial_position) row(s) in presented",
            file="presented.parquet",
        ))

    # Same checks for recalled: output_position uniqueness + list_group
    rpath = dataset_dir / "recalled.parquet"
    if rpath.exists():
        try:
            rdf = pq.read_table(rpath).to_pandas()
        except Exception:
            return v
        dup_r = rdf.duplicated(subset=["participant", "list", "output_position"])
        if dup_r.any():
            v.append(Violation(
                ValidationRule.DUPLICATE_ROW,
                f"{int(dup_r.sum())} duplicate (participant, list, output_position) row(s) in recalled",
                file="recalled.parquet",
            ))
        lg_bad_r = rdf[~(
            ((rdf["list"] < 8) & (rdf["list_group"] == "early"))
            | ((rdf["list"] >= 8) & (rdf["list_group"] == "late"))
        )]
        if len(lg_bad_r) > 0:
            v.append(Violation(
                ValidationRule.LIST_GROUP_MISMATCH,
                f"{len(lg_bad_r)} recalled row(s) have list_group inconsistent with list index",
                file="recalled.parquet", column="list_group",
            ))

        # Rule 3: orphan recalls
        in_list = rdf[rdf["serial_position"] > 0]
        merged = in_list.merge(
            df[["participant", "list", "serial_position"]].assign(_found=1),
            on=["participant", "list", "serial_position"],
            how="left",
        )
        orphans = merged[merged["_found"].isna()]
        if len(orphans) > 0:
            v.append(Violation(
                ValidationRule.ORPHAN_RECALL,
                f"{len(orphans)} recalled row(s) reference a presented row that does not exist",
                file="recalled.parquet", column="serial_position",
            ))
    return v


def validate_dataset(path: str | os.PathLike[str]) -> ValidationReport:
    """Validate a dataset directory against the seven rules in data-model.md §7."""
    dataset_dir = Path(path)
    violations: list[Violation] = []

    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        violations.append(Violation(
            ValidationRule.MISSING_FILE_OR_COLUMN,
            "manifest.json not found",
            file="manifest.json",
        ))
        return ValidationReport(ok=False, violations=violations)

    try:
        manifest = read_json(manifest_path)
    except Exception as exc:
        violations.append(Violation(
            ValidationRule.MANIFEST_MALFORMED,
            f"manifest.json is not valid JSON: {exc!r}",
            file="manifest.json",
        ))
        return ValidationReport(ok=False, violations=violations)

    violations.extend(_check_manifest_shape(manifest))
    violations.extend(_check_files_and_schemas(dataset_dir, manifest))
    violations.extend(_check_hashes(dataset_dir, manifest))
    violations.extend(_check_design_consistency(dataset_dir, manifest))

    return ValidationReport(ok=len(violations) == 0, violations=violations)
