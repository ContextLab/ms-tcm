"""CLI-surface tests for the ``ms-tcm`` entry point.

Covers the two contract items exercised from a shell:
  * ``ms-tcm --help`` and ``ms-tcm fit --help`` print usage and exit 0
    (regression: a prior bug used ``95%`` in a help string, which argparse
    interpreted as a format specifier and crashed with ``ValueError:
    unsupported format character 'b'``).
  * ``ms-tcm fit --alpha`` exits non-zero with a clear message (the §5.2
    conversational-references mechanism requires an edge table that
    free-recall datasets like FRFR-category do not carry).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ms_tcm.cli", *args],
        capture_output=True, text=True, cwd=REPO,
    )


def test_ms_tcm_help_does_not_crash() -> None:
    r = _run(["--help"])
    assert r.returncode == 0, r.stderr
    assert "validate" in r.stdout
    assert "fit" in r.stdout


def test_ms_tcm_fit_help_does_not_crash() -> None:
    r = _run(["fit", "--help"])
    assert r.returncode == 0, r.stderr
    assert "--standard-tcm" in r.stdout
    assert "--alpha" in r.stdout


def test_ms_tcm_fit_alpha_rejected_for_free_recall(tmp_path: Path) -> None:
    r = _run([
        "fit", "data/raw/frfr_category",
        "--out", str(tmp_path / "alpha_rejected"),
        "--alpha", "--n-bootstraps", "1", "--n-restarts", "1", "--force",
    ])
    assert r.returncode != 0
    assert "alpha" in r.stderr.lower() or "5.2" in r.stderr
    # The rejected flag should abort before any output files are written.
    assert not (tmp_path / "alpha_rejected" / "fit_summary.json").exists()


def test_ms_tcm_validate_on_bundled_dataset() -> None:
    r = _run(["validate", "data/raw/frfr_category"])
    assert r.returncode == 0, r.stderr
