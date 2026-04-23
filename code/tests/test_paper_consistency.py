"""Shell-out test for scripts/check_paper_consistency.py (T050 / FR-049).

The script greps paper/main.tex for banned v1 strings and required v6
strings, verifies the CornellZhang2025 bib entry, and verifies the
notes/v6_migration.md has all three required sections. See
specs/002-ms-tcm-v6-hcmr/contracts/paper-consistency.md §9 for the
exit-code policy.

Marked ``slow`` because a proper check (per contract §9 / §3) compiles
the paper first to ensure the checked main.tex is actually built. We
keep the shell-out lightweight — just exec the script on the already-
committed main.tex — because paper compilation on CI runners is covered
by a separate workflow step.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check_paper_consistency.py"


@pytest.mark.slow
def test_check_paper_consistency_exit_zero() -> None:
    """FR-049 + SC-004: the consistency check must exit 0 on the committed
    paper. Banned v1 strings (beta_G, w_G, 0.866, etc.) must be absent;
    required v6 strings (CornellZhang2025, hierarchical context, M^IC,
    M^SC, beta_enc, beta_story, lambda) must be present; the CornZhan25
    bib entry must be present with author/title/journal/year/(doi|url|note).
    """
    assert _SCRIPT.exists(), f"missing check script at {_SCRIPT}"
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=_REPO_ROOT,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"check_paper_consistency.py exited {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
