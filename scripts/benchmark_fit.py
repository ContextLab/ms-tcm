#!/usr/bin/env python3
"""Run a deterministic fit benchmark and append the record to the log.

Per FR-031 and contracts/cli.md ``ms-tcm benchmark``. Emits a one-line JSON
summary to stdout, appends a CSV row to
``data/processed/benchmarks/benchmark_log.csv``, and exits non-zero if the
wall-clock exceeds the tier gate (Tier 1: 120 s; Tier 2: 30 s).

Usage:
    python scripts/benchmark_fit.py data/raw/frfr_category \
        --tier tier1 --seed 42 --n-bootstraps 1000 --n-restarts 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Put code/ on sys.path so the ms_tcm package is importable without install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "code"))

from ms_tcm.benchmark import benchmark_fit  # noqa: E402


_TIER_GATES = {
    "tier1": 120.0,
    "tier2": 30.0,
    "tier2-float64": 30.0,
    "tier2-float32": 30.0,
}


_CSV_COLUMNS = [
    "timestamp", "git_sha", "platform", "tier", "dtype",
    "n_participants", "n_bootstraps", "n_restarts", "seed",
    "wall_clock_seconds", "peak_memory_mb",
]


def _append_csv(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in _CSV_COLUMNS})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", help="Path to a dataset directory.")
    parser.add_argument(
        "--tier", default="tier1",
        choices=["tier1", "tier2", "tier2-float64", "tier2-float32"],
        help="Performance tier (default: tier1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--n-restarts", type=int, default=5)
    parser.add_argument("--n-processes", type=int, default=None)
    parser.add_argument(
        "--standard-tcm", action="store_true",
        help="Benchmark the --standard-tcm reduction instead of MS-TCM.",
    )
    parser.add_argument(
        "--log",
        default=str(_REPO_ROOT / "data" / "processed" / "benchmarks" / "benchmark_log.csv"),
        help="Path to the CSV benchmark log (default: data/processed/benchmarks/benchmark_log.csv).",
    )
    parser.add_argument(
        "--no-gate", action="store_true",
        help="Do not exit non-zero on tier-gate violation (for diagnostics).",
    )
    args = parser.parse_args(argv)

    record = benchmark_fit(
        args.dataset_dir,
        tier=args.tier,
        n_bootstraps=int(args.n_bootstraps),
        n_restarts=int(args.n_restarts),
        seed=int(args.seed),
        n_processes=args.n_processes,
        standard_tcm=bool(args.standard_tcm),
    )

    _append_csv(Path(args.log), record)
    print(json.dumps(record, indent=2, sort_keys=True))

    gate = _TIER_GATES.get(record["tier"], None)
    if gate is not None and record["wall_clock_seconds"] > gate and not args.no_gate:
        print(
            f"error: wall-clock {record['wall_clock_seconds']:.1f} s exceeds "
            f"gate {gate:.1f} s for tier {record['tier']!r} (FR-031).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
