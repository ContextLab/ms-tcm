"""ms-tcm command-line interface.

See contracts/cli.md for the full contract.

Exit codes:
    0 success
    1 argument validation error
    2 input file missing / unreadable
    3 schema validation failed
    4 fit aborted: <90% bootstraps converged
    5 unexpected runtime error
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ms_tcm.bootstrap import bootstrap_ci
from ms_tcm.dataset import load_dataset
from ms_tcm.fit import FitError
from ms_tcm.schema import validate_dataset


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


def _cmd_fit(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out)

    if not dataset_dir.exists():
        print(f"error: dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 2

    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.force:
            print(
                f"error: output directory {out_dir} is not empty; "
                "pass --force to replace",
                file=sys.stderr,
            )
            return 1
        # Remove only ms-tcm fit artifacts.
        for name in ("fit_summary.json", "fit_bootstrap.parquet"):
            candidate = out_dir / name
            if candidate.exists():
                candidate.unlink()

    dataset = load_dataset(dataset_dir)

    optional: dict[str, bool] = {}
    if args.gamma:
        optional["gamma"] = True
    if args.lambda_interference:
        optional["lambda"] = True

    try:
        result = bootstrap_ci(
            dataset,
            n_bootstraps=args.n_bootstraps,
            seed=args.seed,
            n_restarts=args.n_restarts,
            standard_tcm=args.standard_tcm,
            ci=args.ci,
            optional_mechanisms=optional,
            separate_retrieval_weights=args.separate_retrieval_weights,
        )
    except FitError as exc:
        print(f"fit aborted: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:  # noqa: BLE001
        print(f"unexpected error: {exc!r}", file=sys.stderr)
        return 5

    result.save(out_dir)
    print(f"wrote {out_dir}/fit_summary.json and fit_bootstrap.parquet")
    if args.json:
        print(json.dumps(
            {"parameters": result.parameters, "log_likelihood": result.log_likelihood,
             "aic": result.aic, "bic": result.bic,
             "elapsed_seconds": result.elapsed_seconds},
            indent=2, sort_keys=True,
        ))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ms-tcm", description="MS-TCM CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # validate
    v = subparsers.add_parser(
        "validate", help="Validate a dataset directory against the seven rules.",
    )
    v.add_argument("dataset_dir", help="Path to the dataset directory.")
    v.add_argument("--json", action="store_true", help="Emit JSON output.")
    v.set_defaults(func=_cmd_validate)

    # fit
    f = subparsers.add_parser(
        "fit", help="Fit MS-TCM (or standard TCM) with 95% bootstrap CIs.",
    )
    f.add_argument("dataset_dir", help="Path to the dataset directory.")
    f.add_argument("--out", required=True, help="Output directory for fit artifacts.")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--n-bootstraps", type=int, default=1000)
    f.add_argument("--n-restarts", type=int, default=5)
    f.add_argument("--ci", type=float, default=0.95)
    f.add_argument("--standard-tcm", action="store_true",
                   help="Constrain w_storyline=0 for the TCM baseline.")
    f.add_argument("--gamma", action="store_true",
                   help="Enable Section-5.1 resumption reinstatement.")
    f.add_argument("--lambda", dest="lambda_interference", action="store_true",
                   help="Enable Section-5.3 differential interference.")
    f.add_argument("--separate-retrieval-weights", action="store_true",
                   help="Fit w_G^ret independently of w_G.")
    f.add_argument("--force", action="store_true",
                   help="Overwrite an existing non-empty --out directory.")
    f.add_argument("--json", action="store_true",
                   help="Print the fit summary as JSON on stdout after writing to disk.")
    f.set_defaults(func=_cmd_fit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
