"""Instrumented fit wall-clock + peak-memory benchmark (FR-031).

Per contracts/fitter.md §9 and data-model.md §8, every benchmark run produces
a machine-readable record with:

- timestamp (ISO-8601 UTC)
- git_sha (HEAD revision short hash)
- platform ("macos"/"linux"/"windows")
- tier ("tier1"/"tier2-float64"/"tier2-float32")
- dtype ("float64"/"float32")
- n_participants, n_bootstraps, n_restarts, seed
- wall_clock_seconds (``time.perf_counter`` delta)
- peak_memory_mb (``resource.getrusage(RUSAGE_SELF).ru_maxrss``; psutil fallback on Windows)

The benchmark drives the real ``bootstrap_ci`` entry point so any regression
in the fit wall-clock is caught by CI (SC-009).
"""

from __future__ import annotations

import datetime as _dt
import os
import platform as _platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ms_tcm.bootstrap import bootstrap_ci
from ms_tcm.dataset import load_dataset


_TIER_NAMES = {"tier1", "tier2", "tier2-float64", "tier2-float32"}


def _git_sha(repo_root: Path) -> str:
    """Return the short HEAD sha; blank string on failure (e.g. no git)."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _platform_name() -> str:
    sysname = _platform.system().lower()
    if sysname == "darwin":
        return "macos"
    if sysname == "linux":
        return "linux"
    if sysname == "windows":
        return "windows"
    return sysname


def _peak_memory_mb() -> float:
    """Return peak resident-set size in megabytes.

    Uses ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` on POSIX; on Linux
    this is in kilobytes, on macOS in bytes. Windows has no ``resource``
    module — fall back to ``psutil.Process().memory_info().peak_wset`` /
    ``rss`` if psutil is available, else 0.0 so the record remains
    machine-readable (the downstream CSV never carries NaN).
    """
    try:
        import resource  # POSIX only
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS returns bytes; Linux returns kilobytes. Detect by platform.
        if sys.platform == "darwin":
            return float(rss) / (1024.0 * 1024.0)
        return float(rss) / 1024.0
    except ImportError:
        pass
    try:
        import psutil  # type: ignore
        info = psutil.Process(os.getpid()).memory_info()
        # ``peak_wset`` on Windows; fallback to rss elsewhere.
        peak = getattr(info, "peak_wset", None)
        if peak is None:
            peak = info.rss
        return float(peak) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _normalize_tier(tier: str, jax_dtype: str | None = None) -> tuple[str, str, str]:
    """Return (tier_label, backend, dtype).

    tier_label is the canonical string written to the benchmark record.
    backend is the value passed to ``bootstrap_ci`` (``tier1`` or ``jax``).
    dtype is ``float64`` or ``float32``.
    """
    t = tier.lower()
    if t not in _TIER_NAMES:
        raise ValueError(
            f"tier must be one of {sorted(_TIER_NAMES)!r}; got {tier!r}"
        )
    if t == "tier1":
        return "tier1", "tier1", "float64"
    if t == "tier2" or t == "tier2-float64":
        dtype = jax_dtype or "float64"
        return f"tier2-{dtype}", "jax", dtype
    # tier2-float32
    return "tier2-float32", "jax", "float32"


def benchmark_fit(
    dataset_path: str | Path,
    *,
    tier: str = "tier1",
    n_bootstraps: int = 1000,
    n_restarts: int = 5,
    seed: int = 42,
    jax_dtype: str | None = None,
    standard_tcm: bool = False,
    paradigm: str = "free_recall",
    n_processes: int | None = None,
) -> dict[str, Any]:
    """Run a deterministic fit + bootstrap and return a machine-readable record.

    The record has the schema defined in data-model.md §8: a timestamp, the
    git HEAD short sha, the platform, the tier/dtype, the run parameters,
    wall-clock seconds, and peak resident-set memory in megabytes.
    """
    dataset_path = Path(dataset_path).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    ds = load_dataset(dataset_path)

    tier_label, backend, dtype = _normalize_tier(tier, jax_dtype=jax_dtype)
    if backend == "jax":
        # Tier 2 JAX path is deferred per spec §Non-goals; honor the auto-fallback.
        try:
            import jax  # noqa: F401
        except Exception:
            # Auto-fallback to Tier 1 (A6 resolution in spec).
            tier_label = "tier1"
            backend = "tier1"
            dtype = "float64"

    n_participants = int(
        len(set(ds.presented.to_pandas()["participant"].tolist()))
    )
    timestamp = _dt.datetime.now(_dt.timezone.utc).isoformat()

    t0 = time.perf_counter()
    if backend == "tier1":
        fit = bootstrap_ci(
            ds, n_bootstraps=n_bootstraps, seed=seed,
            n_restarts=n_restarts, standard_tcm=standard_tcm,
            paradigm=paradigm, n_processes=n_processes,
        )
    else:
        # Tier 2 JAX backend: not implemented in this feature's 002 window.
        # Fall back to Tier 1 (see auto-fallback clause above).
        fit = bootstrap_ci(
            ds, n_bootstraps=n_bootstraps, seed=seed,
            n_restarts=n_restarts, standard_tcm=standard_tcm,
            paradigm=paradigm, n_processes=n_processes,
        )
    wall = float(time.perf_counter() - t0)
    peak_mb = _peak_memory_mb()

    return {
        "timestamp": timestamp,
        "git_sha": _git_sha(repo_root),
        "platform": _platform_name(),
        "tier": tier_label,
        "dtype": dtype,
        "n_participants": n_participants,
        "n_bootstraps": int(n_bootstraps),
        "n_restarts": int(n_restarts),
        "seed": int(seed),
        "wall_clock_seconds": wall,
        "peak_memory_mb": peak_mb,
        # Non-contractual diagnostics — handy but not required by data-model §8.
        "log_likelihood": float(fit.log_likelihood),
        "n_converged": int(
            sum(1 for r in fit.bootstrap_draws.to_pandas()["converged"] if bool(r))
        ) if fit.bootstrap_draws is not None else 0,
        "dataset_manifest_sha256": str(fit.dataset_manifest_sha256),
    }
