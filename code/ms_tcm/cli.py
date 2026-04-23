"""ms-tcm command-line interface (v6).

See ``specs/002-ms-tcm-v6-hcmr/contracts/cli.md`` for the full contract.

Subcommands:
    validate   - dataset schema/hash validation (unchanged from 001).
    fit        - MLE + bootstrap-CI fit (Tier 1 default; Tier 2 JAX optional).
    benchmark  - deterministic wall-clock benchmark.

Exit codes:
    0   success
    1   wall-clock exceeds tier gate (benchmark only) / fit failure
    2   input validation failure (bad dataset or params)
    3   schema validation failed (``validate`` subcommand)
    130 user interrupt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

from ms_tcm.schema import validate_dataset


# --- validate -------------------------------------------------------------


def _cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.dataset_dir)
    if not path.exists():
        print(f"error: dataset directory not found: {path}", file=sys.stderr)
        return 2
    report = validate_dataset(path)
    if args.json:
        payload = {
            "ok": report.ok,
            "violations": [
                {
                    "rule": v.rule.name,
                    "message": v.message,
                    "file": v.file,
                    "column": v.column,
                    "row": v.row,
                }
                for v in report.violations
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(report.summary())
    return 0 if report.ok else 3


# --- fit ------------------------------------------------------------------


def _resolve_backend(args: argparse.Namespace) -> str:
    """Return the effective backend: CLI flag > env var > built-in default."""
    if getattr(args, "backend", None):
        return str(args.backend).lower()
    env = os.environ.get("MS_TCM_BACKEND", "").lower()
    if env:
        return env
    return "tier1"


def _resolve_jax_dtype(args: argparse.Namespace) -> str:
    """Return the effective JAX dtype: CLI flag > env var > float64 default."""
    if getattr(args, "jax_dtype", None):
        return str(args.jax_dtype).lower()
    env = os.environ.get("MS_TCM_JAX_DTYPE", "").lower()
    if env:
        return env
    return "float64"


def _write_fit_outputs(fit_result, out_dir: Path) -> None:
    """Write fit_summary.json + fit_bootstrap.parquet for a FitResult."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "parameters": fit_result.parameters,
        "log_likelihood": float(fit_result.log_likelihood),
        "aic": float(fit_result.aic),
        "bic": float(fit_result.bic),
        "n_participants": int(fit_result.n_participants),
        "n_lists": int(fit_result.n_lists),
        "n_recalls_used": int(fit_result.n_recalls_used),
        "n_intrusions_excluded": int(fit_result.n_intrusions_excluded),
        "seed": int(fit_result.seed),
        "elapsed_seconds": float(fit_result.elapsed_seconds),
        "ms_tcm_version": str(fit_result.ms_tcm_version),
        "dataset_manifest_sha256": str(fit_result.dataset_manifest_sha256),
        "standard_tcm": bool(fit_result.standard_tcm),
        "backend": str(fit_result.backend),
    }
    (out_dir / "fit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if fit_result.bootstrap_draws is not None:
        import pyarrow.parquet as pq
        pq.write_table(
            fit_result.bootstrap_draws, out_dir / "fit_bootstrap.parquet",
            compression="none",
        )


def _cmd_fit(args: argparse.Namespace) -> int:
    """MLE + bootstrap-CI fit per contracts/cli.md ``ms-tcm fit``."""
    path = Path(args.dataset_dir)
    if not path.exists():
        print(f"error: dataset directory not found: {path}", file=sys.stderr)
        return 2
    if not args.out and not args.dry_run:
        print("error: --out is required unless --dry-run is set", file=sys.stderr)
        return 2

    backend = _resolve_backend(args)
    if backend == "jax":
        try:
            import jax  # noqa: F401
        except Exception:
            warnings.warn(
                "JAX backend requested but `jax` is not importable; "
                "falling back to Tier 1. Install with `pip install -e .[jax]`."
            )
            backend = "tier1"

    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "dataset_dir": str(path),
            "out": args.out,
            "seed": int(args.seed),
            "n_restarts": int(args.n_restarts),
            "n_bootstraps": int(args.n_bootstraps),
            "paradigm": args.paradigm,
            "standard_tcm": bool(args.standard_tcm),
            "backend": backend,
            "jax_dtype": _resolve_jax_dtype(args) if backend == "jax" else "float64",
            "pre_context": args.pre_context,
        }, indent=2, sort_keys=True))
        return 0

    # Validate the --pre-context value early so a bogus path fails before
    # the fit starts. "identity" is the default and requires no setup;
    # any other value is interpreted as a Parquet path and must exist.
    # Non-identity loading currently raises NotImplementedError per T011
    # (USE integration is out of scope for feature 002 / spec §Non-goals).
    pre_context = str(args.pre_context)
    if pre_context != "identity":
        pc_path = Path(pre_context)
        if not pc_path.exists():
            print(
                f"error: --pre-context path does not exist: {pc_path}",
                file=sys.stderr,
            )
            return 2
        # Load via EmbeddingPreMatrix.from_parquet — explicitly raises
        # NotImplementedError today; when T011 lands USE support, this
        # call site will start working without any CLI changes.
        from ms_tcm.preexp import EmbeddingPreMatrix

        try:
            EmbeddingPreMatrix.from_parquet(pc_path)
        except NotImplementedError as exc:
            print(
                f"error: --pre-context=<path> is not yet supported: {exc}",
                file=sys.stderr,
            )
            return 2

    from ms_tcm.bootstrap import bootstrap_ci
    from ms_tcm.dataset import load_dataset

    ds = load_dataset(path)
    fit = bootstrap_ci(
        ds,
        n_bootstraps=int(args.n_bootstraps),
        seed=int(args.seed),
        n_restarts=int(args.n_restarts),
        standard_tcm=bool(args.standard_tcm),
        paradigm=str(args.paradigm),
        n_processes=args.n_processes,
    )
    _write_fit_outputs(fit, Path(args.out))
    print(json.dumps({
        "ok": True,
        "out": str(args.out),
        "log_likelihood": float(fit.log_likelihood),
        "elapsed_seconds": float(fit.elapsed_seconds),
        "backend": backend,
    }, indent=2, sort_keys=True))
    return 0


# --- benchmark ------------------------------------------------------------


_BENCHMARK_GATES = {
    "tier1": 120.0, "tier2": 30.0,
    "tier2-float64": 30.0, "tier2-float32": 30.0,
}


_BENCHMARK_CSV_COLUMNS = [
    "timestamp", "git_sha", "platform", "tier", "dtype",
    "n_participants", "n_bootstraps", "n_restarts", "seed",
    "wall_clock_seconds", "peak_memory_mb",
]


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Deterministic benchmark; appends CSV row; exits non-zero on gate breach."""
    path = Path(args.dataset_dir)
    if not path.exists():
        print(f"error: dataset directory not found: {path}", file=sys.stderr)
        return 2

    from ms_tcm.benchmark import benchmark_fit

    record = benchmark_fit(
        str(path),
        tier=args.tier,
        n_bootstraps=int(args.n_bootstraps),
        n_restarts=int(args.n_restarts),
        seed=int(args.seed),
        n_processes=args.n_processes,
        standard_tcm=bool(args.standard_tcm),
    )

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_BENCHMARK_CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in _BENCHMARK_CSV_COLUMNS})

    print(json.dumps(record, indent=2, sort_keys=True))
    gate = _BENCHMARK_GATES.get(record["tier"], None)
    if gate is not None and record["wall_clock_seconds"] > gate and not args.no_gate:
        print(
            f"error: wall-clock {record['wall_clock_seconds']:.1f} s exceeds "
            f"gate {gate:.1f} s for tier {record['tier']!r} (FR-031).",
            file=sys.stderr,
        )
        return 1
    return 0


# --- Argument parser ------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    from ms_tcm import __version__

    parser = argparse.ArgumentParser(prog="ms-tcm", description="MS-TCM CLI (v6)")
    parser.add_argument("--version", "-V", action="version", version=str(__version__))
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # validate
    v = subparsers.add_parser(
        "validate", help="Validate a dataset directory against the seven rules.",
    )
    v.add_argument("dataset_dir", help="Path to the dataset directory.")
    v.add_argument("--json", action="store_true", help="Emit JSON output.")
    v.set_defaults(func=_cmd_validate)

    # fit
    f = subparsers.add_parser("fit", help="MLE + bootstrap-CI fit.")
    f.add_argument("dataset_dir", help="Path to a dataset directory.")
    f.add_argument("--out", help="Output directory for fit_summary.json + fit_bootstrap.parquet.")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--n-restarts", type=int, default=5)
    f.add_argument("--n-bootstraps", type=int, default=1000)
    f.add_argument("--paradigm", choices=["free_recall", "cued_recall"], default="free_recall")
    f.add_argument("--standard-tcm", action="store_true")
    f.add_argument(
        "--pre-context", default="identity",
        help='"identity" (default) or a path to an embeddings Parquet.',
    )
    f.add_argument("--backend", default=None, choices=[None, "tier1", "jax"])
    f.add_argument("--jax-dtype", default=None, choices=[None, "float64", "float32"])
    f.add_argument("--n-processes", type=int, default=None)
    f.add_argument(
        "--force", action="store_true",
        help="Overwrite existing outputs (legacy flag kept for CI compatibility).",
    )
    f.add_argument("--dry-run", action="store_true")
    f.set_defaults(func=_cmd_fit)

    # benchmark
    b = subparsers.add_parser("benchmark", help="Deterministic wall-clock benchmark.")
    b.add_argument("dataset_dir")
    b.add_argument(
        "--tier", default="tier1",
        choices=["tier1", "tier2", "tier2-float64", "tier2-float32"],
    )
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--n-bootstraps", type=int, default=1000)
    b.add_argument("--n-restarts", type=int, default=5)
    b.add_argument("--n-processes", type=int, default=None)
    b.add_argument("--standard-tcm", action="store_true")
    b.add_argument(
        "--log",
        default="data/processed/benchmarks/benchmark_log.csv",
        help="Path to CSV benchmark log.",
    )
    b.add_argument("--no-gate", action="store_true")
    b.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
