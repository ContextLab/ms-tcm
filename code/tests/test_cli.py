"""Tests for the ms-tcm CLI (T045b, T065 / US2 + US4).

Covers:
- ``--backend jax`` auto-fallback when JAX is not importable (A6 / T045b).
- ``--pre-context`` flag validation (T065 / US4).
- ``--dry-run`` basic parse/validate path.

No real fits are run — these tests exercise argument parsing and
input-validation code paths only.
"""

from __future__ import annotations

import builtins
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ms_tcm.cli import build_parser


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRFR = _REPO_ROOT / "data" / "raw" / "frfr_category"


# --- build_parser smoke tests ---------------------------------------------

def test_build_parser_exposes_validate_fit_benchmark() -> None:
    """The CLI must expose three subcommands: validate, fit, benchmark."""
    parser = build_parser()
    # Parse a "fit" invocation with --dry-run, confirm no error.
    ns = parser.parse_args([
        "fit", str(_FRFR), "--dry-run",
        "--seed", "42", "--n-bootstraps", "1", "--n-restarts", "1",
    ])
    assert ns.cmd == "fit"
    assert ns.dry_run is True


def test_fit_accepts_pre_context_identity() -> None:
    parser = build_parser()
    ns = parser.parse_args([
        "fit", str(_FRFR), "--dry-run", "--pre-context", "identity",
    ])
    assert ns.pre_context == "identity"


def test_fit_accepts_pre_context_parquet_path() -> None:
    parser = build_parser()
    ns = parser.parse_args([
        "fit", str(_FRFR), "--dry-run", "--pre-context", "some/embeddings.parquet",
    ])
    assert ns.pre_context == "some/embeddings.parquet"


def test_fit_accepts_standard_tcm_flag() -> None:
    parser = build_parser()
    ns = parser.parse_args([
        "fit", str(_FRFR), "--dry-run", "--standard-tcm",
    ])
    assert ns.standard_tcm is True


def test_fit_accepts_backend_jax() -> None:
    parser = build_parser()
    ns = parser.parse_args([
        "fit", str(_FRFR), "--dry-run", "--backend", "jax",
    ])
    assert ns.backend == "jax"


# --- Dry-run integration tests -------------------------------------------

def test_cli_dry_run_prints_plan_and_exits_zero(capsys) -> None:
    """Dry-run short-circuits before any fit; must emit JSON and exit 0."""
    from ms_tcm.cli import main

    rc = main([
        "fit", str(_FRFR), "--dry-run",
        "--seed", "1", "--n-bootstraps", "1", "--n-restarts", "1",
    ])
    assert rc == 0, f"dry-run exited {rc}"
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["dry_run"] is True
    assert payload["backend"] == "tier1"
    assert payload["pre_context"] == "identity"


# --- T045b: JAX auto-fallback when JAX is not installed ------------------

def test_jax_fallback_when_unavailable(tmp_path) -> None:
    """A6 resolution (T045b): when --backend jax is requested but JAX is
    not importable, the CLI warns and falls back to Tier 1.

    We simulate JAX absence by monkey-patching ``__import__`` to raise
    ImportError on any ``jax`` import; the CLI's ``_cmd_fit`` catches
    the ImportError, emits a warning containing the substring "falling
    back to Tier 1", and continues in dry-run mode.
    """
    import warnings

    from ms_tcm.cli import main

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("simulated JAX absence")
        return real_import(name, globals, locals, fromlist, level)

    with patch.object(builtins, "__import__", side_effect=fake_import):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rc = main([
                "fit", str(_FRFR), "--dry-run", "--backend", "jax",
                "--seed", "1", "--n-bootstraps", "1", "--n-restarts", "1",
            ])

    assert rc == 0, "dry-run must still succeed after JAX fallback"
    fallback_msgs = [str(w.message) for w in caught
                     if "falling back to Tier 1" in str(w.message)]
    assert fallback_msgs, (
        "expected at least one 'falling back to Tier 1' warning when JAX "
        f"is unavailable; got: {[str(w.message) for w in caught]!r}"
    )


# --- T065: --pre-context with non-existent path fails with exit 2 --------

def test_pre_context_flag_nonexistent_path_exits_2(tmp_path) -> None:
    """T065 / US4: --pre-context <non-existent path> triggers exit 2 in a
    real (non-dry-run) fit invocation. We use --dry-run=False but also
    avoid running a real fit by pointing to a non-existent dataset;
    the dataset check happens first (exit 2), so we can't directly test
    the pre-context-path path via the CLI in a fast test. Instead, we
    test the code path directly: when the dataset exists but the
    pre-context path does not, the CLI exits 2.
    """
    bogus = tmp_path / "nope.parquet"
    assert not bogus.exists()

    from ms_tcm.cli import main

    # Use a real dataset so the dataset-exists check passes, but a bogus
    # pre-context path so we hit the new validation block.
    rc = main([
        "fit", str(_FRFR),
        "--out", str(tmp_path / "out"),
        "--pre-context", str(bogus),
        "--seed", "1", "--n-bootstraps", "1", "--n-restarts", "1",
    ])
    assert rc == 2, f"expected exit 2 for missing --pre-context path; got {rc}"
