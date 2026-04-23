"""ms-tcm command-line interface (Phase-1 window).

See ``specs/002-ms-tcm-v6-hcmr/contracts/cli.md`` for the full contract. The
``fit`` / ``run`` / ``benchmark`` subcommands land in Phase 4 and Phase 5 of
feature 002; during Phase 1 only ``validate`` is available.

Exit codes:
    0 success
    1 argument validation error
    2 input file missing / unreadable
    3 schema validation failed
    4 fit aborted: <90% bootstraps converged (Phase 4+)
    5 unexpected runtime error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def _cmd_fit_stub(args: argparse.Namespace) -> int:
    print(
        "error: 'ms-tcm fit' is disabled during the v6 Phase-1 window. "
        "The v6 fitter lands in Phase 4 of feature 002-ms-tcm-v6-hcmr. "
        "See specs/002-ms-tcm-v6-hcmr/tasks.md (T031, T032, T041).",
        file=sys.stderr,
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ms-tcm", description="MS-TCM CLI (v6)")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # validate
    v = subparsers.add_parser(
        "validate", help="Validate a dataset directory against the seven rules.",
    )
    v.add_argument("dataset_dir", help="Path to the dataset directory.")
    v.add_argument("--json", action="store_true", help="Emit JSON output.")
    v.set_defaults(func=_cmd_validate)

    # fit (stub — real subcommand lands in Phase 4)
    f = subparsers.add_parser(
        "fit", help="[disabled in Phase 1] MLE + bootstrap CI fit.",
    )
    f.add_argument("dataset_dir", nargs="?")
    f.add_argument("--out", required=False)
    f.set_defaults(func=_cmd_fit_stub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
