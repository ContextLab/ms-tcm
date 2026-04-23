#!/usr/bin/env python3
"""Build reference empirical SPC, pFR, and lag-CRP curves for behavioral regression.

Per FR-020 Layer 2 (`specs/002-ms-tcm-v6-hcmr/contracts/regression-tests.md §6`):
Reads the FRFR-category dataset, computes empirical SPC / pFR / lag-CRP via
`code/analyses/{spc,pfr,lag_crp}.py`, writes Parquet files to
`data/processed/reference_curves/`, plus a companion `manifest.json` with
SHA-256s.

Idempotent: same input => byte-identical output.

Usage:
    python scripts/build_reference_curves.py data/raw/frfr_category \
        --out data/processed/reference_curves/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Put code/ on sys.path so `ms_tcm` and `analyses` are importable without install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "code"))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ms_tcm.dataset import Dataset, load_dataset  # noqa: E402
from analyses.spc import compute_spc  # noqa: E402
from analyses.pfr import compute_pfr  # noqa: E402
from analyses.lag_crp import compute_lag_crp  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _series_to_table(series: pd.Series, index_name: str, value_name: str) -> pa.Table:
    """Convert a pandas Series to a deterministic pyarrow Table with explicit dtypes."""
    # Drop NaN-valued entries (e.g. lag=0 in lag-CRP always returns NaN).
    clean = series.dropna()
    return pa.table({
        index_name: pa.array(clean.index.to_numpy().astype("int64"), type=pa.int64()),
        value_name: pa.array(clean.to_numpy().astype("float64"), type=pa.float64()),
    })


def build_reference_curves(dataset_path: Path, out_dir: Path) -> dict:
    """Build and write the three reference curves; return a manifest dict."""
    ds = load_dataset(dataset_path)
    W = ds.num_words_per_list

    spc_series = compute_spc(ds)
    pfr_series = compute_pfr(ds)
    crp_series = compute_lag_crp(ds)

    spc_tbl = _series_to_table(spc_series, "serial_position", "p_recall")
    pfr_tbl = _series_to_table(pfr_series, "serial_position", "p_first_recall")
    crp_tbl = _series_to_table(crp_series, "lag", "crp")

    out_dir.mkdir(parents=True, exist_ok=True)
    spc_path = out_dir / "frfr_category_spc.parquet"
    pfr_path = out_dir / "frfr_category_pfr.parquet"
    crp_path = out_dir / "frfr_category_lag_crp.parquet"

    # Write deterministically: sort by index, no compression metadata drift.
    pq.write_table(
        spc_tbl.sort_by("serial_position"), spc_path, compression="none",
    )
    pq.write_table(
        pfr_tbl.sort_by("serial_position"), pfr_path, compression="none",
    )
    pq.write_table(
        crp_tbl.sort_by("lag"), crp_path, compression="none",
    )

    # Source dataset manifest hash (from the raw FRFR-category manifest.json).
    raw_manifest_path = dataset_path / "manifest.json"
    raw_manifest_sha = _sha256(raw_manifest_path)

    manifest = {
        "source_dataset": str(dataset_path.relative_to(_REPO_ROOT))
            if dataset_path.is_absolute() else str(dataset_path),
        "source_manifest_sha256": raw_manifest_sha,
        "curves": {
            "spc": {
                "file": spc_path.name,
                "sha256": _sha256(spc_path),
                "rows": spc_tbl.num_rows,
                "schema": {"serial_position": "int64", "p_recall": "float64"},
            },
            "pfr": {
                "file": pfr_path.name,
                "sha256": _sha256(pfr_path),
                "rows": pfr_tbl.num_rows,
                "schema": {"serial_position": "int64", "p_first_recall": "float64"},
            },
            "lag_crp": {
                "file": crp_path.name,
                "sha256": _sha256(crp_path),
                "rows": crp_tbl.num_rows,
                "schema": {"lag": "int64", "crp": "float64"},
            },
        },
        "num_words_per_list": int(W),
        "invocation": (
            f"python scripts/build_reference_curves.py {dataset_path} "
            f"--out {out_dir}"
        ),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", help="Path to a dataset directory (e.g. data/raw/frfr_category).")
    parser.add_argument(
        "--out", default="data/processed/reference_curves/",
        help="Output directory for reference curve Parquet files.",
    )
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out).resolve()

    manifest = build_reference_curves(dataset_path, out_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
