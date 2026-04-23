"""Performance regression tests for Tier 1 (+ optional Tier 2) fits.

Spec: specs/002-ms-tcm-v6-hcmr/spec.md FR-031, SC-002.
Contract: specs/002-ms-tcm-v6-hcmr/contracts/fitter.md §9.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRFR = _REPO_ROOT / "data" / "raw" / "frfr_category"


@pytest.mark.slow
def test_tier1_wall_clock_under_120s() -> None:
    """FR-031 / SC-002: Tier 1 fit must complete in <120 s wall-clock for the
    standard invocation on the bundled FRFR-category dataset.

    Runs a full benchmark fit (1000 bootstraps × 5 restarts × seed=42). This
    test is marked ``slow`` because the mandatory run is a few minutes even
    under Tier 1; the CI benchmark job invokes ``scripts/benchmark_fit.py``
    which applies the same assertion.
    """
    from ms_tcm.benchmark import benchmark_fit

    result = benchmark_fit(
        str(_FRFR),
        tier="tier1",
        n_bootstraps=1000,
        n_restarts=5,
        seed=42,
    )
    assert result["wall_clock_seconds"] < 120.0, (
        f"Tier 1 wall-clock {result['wall_clock_seconds']:.1f} s exceeds 120 s "
        f"gate (FR-031). Benchmark record: {result!r}."
    )


@pytest.mark.slow
def test_tier2_wall_clock_under_30s() -> None:
    """FR-032: Tier 2 (JAX) fit must complete in <30 s wall-clock.

    Skipped when ``jax`` is not importable (JAX is an optional extras group).
    """
    jax = pytest.importorskip("jax")  # noqa: F841
    from ms_tcm.benchmark import benchmark_fit

    result = benchmark_fit(
        str(_FRFR),
        tier="tier2",
        n_bootstraps=1000,
        n_restarts=5,
        seed=42,
    )
    assert result["wall_clock_seconds"] < 30.0, (
        f"Tier 2 wall-clock {result['wall_clock_seconds']:.1f} s exceeds 30 s "
        f"gate (FR-032). Benchmark record: {result!r}."
    )


def test_benchmark_record_has_required_fields() -> None:
    """Data-model §8: every benchmark record has timestamp, git_sha, platform,
    tier, dtype, n_participants, n_bootstraps, n_restarts, seed,
    wall_clock_seconds, peak_memory_mb."""
    from ms_tcm.benchmark import benchmark_fit

    result = benchmark_fit(
        str(_FRFR),
        tier="tier1",
        n_bootstraps=2,
        n_restarts=1,
        seed=0,
    )
    required = {
        "timestamp", "git_sha", "platform", "tier", "dtype",
        "n_participants", "n_bootstraps", "n_restarts", "seed",
        "wall_clock_seconds", "peak_memory_mb",
    }
    assert required.issubset(result.keys()), (
        f"benchmark record missing fields: {required - set(result.keys())}"
    )
    assert result["tier"] == "tier1"
    assert result["dtype"] == "float64"
    assert isinstance(result["wall_clock_seconds"], float)
    assert result["wall_clock_seconds"] > 0.0
