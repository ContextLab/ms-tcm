# Contract: Dataset Schema (v6)

**Feature**: 002-ms-tcm-v6-hcmr
**Status**: re-published from [001 contracts/dataset-schema.md](../../001-ms-tcm-impl/contracts/dataset-schema.md) with no changes

The Parquet + manifest dataset schema is **unchanged** from 001. FR-061 preserves the bundled FRFR-category dataset byte-identically; no migration is required.

This file exists so downstream contracts (`regression-tests.md`, `cli.md`) can link to a path inside `specs/002-ms-tcm-v6-hcmr/contracts/` without crossing feature boundaries.

## Forward pointer

Refer to [../../001-ms-tcm-impl/contracts/dataset-schema.md](../../001-ms-tcm-impl/contracts/dataset-schema.md) for:

- `presented.parquet` columns and types
- `recalled.parquet` columns and types
- `manifest.json` top-level keys (`source`, `design`, `files`, `row_counts`, `created_at`)
- The seven validation rules (FR-012 from 001).

## v6 additions

v6 introduces **one new category of derived Parquet artifacts** that share the schema conventions above:

1. `data/processed/reference_curves/frfr_category_spc.parquet` — `(serial_position: int64, p_recall: float64)`.
2. `data/processed/reference_curves/frfr_category_pfr.parquet` — `(serial_position: int64, p_first_recall: float64)`.
3. `data/processed/reference_curves/frfr_category_lag_crp.parquet` — `(lag: int64, crp: float64)`, with the row at `lag=0` omitted.
4. `data/processed/reference_curves/manifest.json` — SHA-256 per file, source dataset's manifest hash, `scripts/build_reference_curves.py` invocation line, creation timestamp.

These are derived artifacts (Constitution II) — regenerable via `python scripts/build_reference_curves.py data/raw/frfr_category --out data/processed/reference_curves/`. The derived manifest's hashes are verified by `test_frfr.py::test_reference_curves_match_manifest`.

## v6 changes to the raw dataset

**None.** Per FR-061, `data/raw/frfr_category/` ships byte-identically. The feature encoder (`code/ms_tcm/features.py`) is vectorized (FR-030) but its output is bit-for-bit unchanged on the same input.
