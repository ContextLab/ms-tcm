"""Participant-level percentile bootstrap for v6 hierarchical CMR.

Per FR-030 / contracts/fitter.md §6: the bootstrap loop is parallelized via
``multiprocessing.Pool``. Workers receive a serializable work-item composed
of numpy arrays and primitives — the pyarrow-backed ``Dataset`` object is
NOT serialized across the process boundary; each worker rebuilds its
Dataset from the per-draw participant-resample on the fly.

Deterministic-equivalence to the serial path is verified by
``test_bootstrap.py::test_parallel_matches_serial_same_seed``.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from ms_tcm import __version__
from ms_tcm.dataset import Dataset
from ms_tcm.fit import FitError, FitResult, fit_mle


@dataclass(frozen=True)
class _BootstrapWorkItem:
    """Arrays + primitives a worker needs to reconstruct a resampled Dataset.

    We pass the numpy-level view of presented + recalled plus the participant
    index remap, so the worker can build its own pyarrow Tables without
    copying a pyarrow-backed Dataset across processes (contracts/fitter.md §6).
    """
    presented_arrays: dict[str, np.ndarray]
    recalled_arrays: dict[str, np.ndarray]
    manifest_str: str
    participant_sample: np.ndarray
    bootstrap_id: int
    seed: int
    n_restarts: int
    standard_tcm: bool
    paradigm: str


def _rebuild_resampled_dataset(work: _BootstrapWorkItem) -> Dataset:
    """Reconstruct a Dataset from a worker's work item + participant resample."""
    pres_df = pd.DataFrame(work.presented_arrays)
    rec_df = pd.DataFrame(work.recalled_arrays)
    pres_groups = {int(p): g for p, g in pres_df.groupby("participant")}
    rec_groups = {int(p): g for p, g in rec_df.groupby("participant")}

    pres_parts: list[pd.DataFrame] = []
    rec_parts: list[pd.DataFrame] = []
    for new_p, old_p in enumerate(work.participant_sample.tolist()):
        pg = pres_groups.get(int(old_p))
        rg = rec_groups.get(int(old_p))
        if pg is None:
            raise FitError(
                f"bootstrap draw {work.bootstrap_id}: participant {old_p!r} "
                f"not found in presented table"
            )
        pg = pg.copy()
        pg["participant"] = int(new_p)
        pres_parts.append(pg)
        if rg is not None:
            rg = rg.copy()
            rg["participant"] = int(new_p)
            rec_parts.append(rg)

    pres_concat = pd.concat(pres_parts, ignore_index=True)
    rec_concat = (
        pd.concat(rec_parts, ignore_index=True)
        if rec_parts else pd.DataFrame(columns=rec_df.columns)
    )
    return Dataset(
        presented=pa.Table.from_pandas(pres_concat, preserve_index=False),
        recalled=pa.Table.from_pandas(rec_concat, preserve_index=False),
        manifest={"__str__": work.manifest_str},
    )


def _run_one_bootstrap(work: _BootstrapWorkItem) -> dict:
    """Run a single bootstrap draw: resample participants, fit MLE."""
    try:
        ds = _rebuild_resampled_dataset(work)
        fit = fit_mle(
            ds,
            n_restarts=work.n_restarts,
            seed=work.seed,
            standard_tcm=work.standard_tcm,
            paradigm=work.paradigm,
        )
        row = {"bootstrap_id": work.bootstrap_id, "converged": True}
        for name, entry in fit.parameters.items():
            row[name] = float(entry["mle"])
        return row
    except Exception as exc:  # pragma: no cover
        return {
            "bootstrap_id": work.bootstrap_id,
            "converged": False,
            "error": str(exc),
        }


def bootstrap_ci(
    dataset: Dataset,
    *,
    n_bootstraps: int = 1000,
    seed: int = 0,
    n_restarts: int = 5,
    standard_tcm: bool = False,
    paradigm: str = "free_recall",
    ci: float = 0.95,
    n_processes: int | None = None,
) -> FitResult:
    """Participant-level percentile bootstrap (parallelized via Pool).

    Returns a ``FitResult`` with ``parameters[name]["ci_lower"]`` /
    ``["ci_upper"]`` populated per parameter, and ``bootstrap_draws`` set to
    the per-draw pyarrow Table.

    Raises ``FitError`` if fewer than 90 % of bootstrap draws converge
    (per contracts/fitter.md §5).
    """
    t0 = time.perf_counter()

    # First: the canonical point-estimate fit on the full dataset.
    point_fit = fit_mle(
        dataset, n_restarts=n_restarts, seed=seed,
        standard_tcm=standard_tcm, paradigm=paradigm,
    )

    # Enumerate participants for resampling.
    pres_df = dataset.presented.to_pandas()
    rec_df = dataset.recalled.to_pandas()
    participants = sorted(pres_df["participant"].unique().tolist())
    n_parts = len(participants)
    if n_parts == 0:
        raise FitError("dataset has no participants; cannot bootstrap")

    presented_arrays = {col: pres_df[col].to_numpy() for col in pres_df.columns}
    recalled_arrays = {col: rec_df[col].to_numpy() for col in rec_df.columns}
    manifest_str = str(dataset.manifest)

    # Deterministic per-draw seeds via SeedSequence spawning.
    master = np.random.SeedSequence(seed)
    children = master.spawn(int(n_bootstraps))

    work_items: list[_BootstrapWorkItem] = []
    for b, child in enumerate(children):
        rng = np.random.default_rng(child)
        sample = rng.choice(participants, size=n_parts, replace=True)
        draw_fit_seed = int(child.generate_state(1)[0]) % (2 ** 31 - 1)
        work_items.append(_BootstrapWorkItem(
            presented_arrays=presented_arrays,
            recalled_arrays=recalled_arrays,
            manifest_str=manifest_str,
            participant_sample=sample,
            bootstrap_id=b,
            seed=draw_fit_seed,
            n_restarts=int(n_restarts),
            standard_tcm=bool(standard_tcm),
            paradigm=str(paradigm),
        ))

    # Dispatch: parallel if n_processes > 1, else serial (for determinism tests).
    if n_processes is None:
        n_processes = max(1, min(len(work_items), mp.cpu_count()))
    if n_processes <= 1:
        rows = [_run_one_bootstrap(w) for w in work_items]
    else:
        with mp.get_context("spawn").Pool(processes=n_processes) as pool:
            rows = list(pool.imap_unordered(_run_one_bootstrap, work_items))
        rows.sort(key=lambda r: r["bootstrap_id"])

    n_converged = int(sum(1 for r in rows if r.get("converged", False)))
    if n_converged < 0.9 * n_bootstraps:
        raise FitError(
            f"bootstrap convergence below threshold: {n_converged}/{n_bootstraps} "
            f"(required >= 90%)"
        )

    columns: dict[str, list] = {"bootstrap_id": [], "converged": []}
    for row in rows:
        columns["bootstrap_id"].append(int(row["bootstrap_id"]))
        columns["converged"].append(bool(row.get("converged", False)))
    for name in point_fit.parameters.keys():
        columns[name] = []
        for row in rows:
            columns[name].append(float(row.get(name, float("nan"))))
    bootstrap_tbl = pa.table(columns)

    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    parameters = dict(point_fit.parameters)
    converged_mask = np.array(columns["converged"], dtype=bool)
    for name, entry in parameters.items():
        draws = np.array(columns[name], dtype=np.float64)
        if converged_mask.any():
            use = draws[converged_mask]
            finite = use[np.isfinite(use)]
        else:
            finite = np.array([], dtype=np.float64)
        new_entry = dict(entry)
        if finite.size:
            new_entry["ci_lower"] = float(np.quantile(finite, lo_q))
            new_entry["ci_upper"] = float(np.quantile(finite, hi_q))
        new_entry["ci_method"] = "percentile"
        new_entry["n_bootstraps"] = int(n_bootstraps)
        new_entry["n_converged"] = int(n_converged)
        parameters[name] = new_entry

    return FitResult(
        parameters=parameters,
        log_likelihood=point_fit.log_likelihood,
        aic=point_fit.aic,
        bic=point_fit.bic,
        n_participants=point_fit.n_participants,
        n_lists=point_fit.n_lists,
        n_recalls_used=point_fit.n_recalls_used,
        n_intrusions_excluded=point_fit.n_intrusions_excluded,
        seed=int(seed),
        elapsed_seconds=float(time.perf_counter() - t0),
        ms_tcm_version=str(__version__),
        dataset_manifest_sha256=point_fit.dataset_manifest_sha256,
        standard_tcm=bool(standard_tcm),
        backend="tier1",
        bootstrap_draws=bootstrap_tbl,
    )
