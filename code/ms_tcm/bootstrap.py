"""Participant-level bootstrap for 95% confidence intervals.

See contracts/fitter.md section 5.
"""

from __future__ import annotations

import hashlib
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from ms_tcm import __version__
from ms_tcm.dataset import Dataset
from ms_tcm.fit import (
    FitError, FitResult, _compute_aic_bic, _free_parameter_names,
    _theta_to_params, fit_mle,
)
from ms_tcm.likelihood import dataset_log_likelihood, likelihood_diagnostics


def _resample_participants(
    dataset: Dataset, indices: np.ndarray,
) -> Dataset:
    """Build a Dataset view from a list of participant ids (with replacement)."""
    pdf = dataset.presented.to_pandas()
    rdf = dataset.recalled.to_pandas()
    pres_parts: list[pd.DataFrame] = []
    rec_parts: list[pd.DataFrame] = []
    for new_pid, old_pid in enumerate(indices):
        p_sub = pdf[pdf["participant"] == int(old_pid)].copy()
        r_sub = rdf[rdf["participant"] == int(old_pid)].copy()
        p_sub["participant"] = new_pid
        r_sub["participant"] = new_pid
        pres_parts.append(p_sub)
        rec_parts.append(r_sub)
    pres = pa.Table.from_pandas(
        pd.concat(pres_parts, ignore_index=True), preserve_index=False,
    )
    rec = pa.Table.from_pandas(
        pd.concat(rec_parts, ignore_index=True), preserve_index=False,
    )
    return Dataset(presented=pres, recalled=rec, manifest=dict(dataset.manifest))


def bootstrap_ci(
    dataset: Dataset,
    *,
    n_bootstraps: int = 1000,
    seed: int,
    n_restarts: int = 5,
    standard_tcm: bool = False,
    ci: float = 0.95,
    optional_mechanisms: dict[str, bool] | None = None,
    separate_retrieval_weights: bool = False,
) -> FitResult:
    """Fit the point MLE + percentile participant-level bootstrap CIs."""
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1); got {ci!r}")

    start_time = perf_counter()
    free_names = _free_parameter_names(
        standard_tcm=standard_tcm,
        optional_mechanisms=optional_mechanisms,
        separate_retrieval_weights=separate_retrieval_weights,
    )

    # Split the seed into two independent streams: one for the point-MLE
    # restarts, one for the bootstrap resampling.
    master = np.random.SeedSequence(seed)
    mle_stream_seed = int(master.spawn(2)[0].generate_state(1)[0])
    boot_stream_seed = int(master.spawn(2)[1].generate_state(1)[0])

    # Point MLE.
    mle_theta = fit_mle(
        dataset, n_restarts=n_restarts, seed=mle_stream_seed,
        standard_tcm=standard_tcm,
        optional_mechanisms=optional_mechanisms,
        separate_retrieval_weights=separate_retrieval_weights,
    )
    params_mle = _theta_to_params(mle_theta)
    log_like_mle = dataset_log_likelihood(dataset, params_mle)

    diag = likelihood_diagnostics(dataset)
    aic, bic = _compute_aic_bic(
        log_like_mle, k=len(free_names), n_recalls_used=diag.n_recalls_used,
    )

    # Participants for resampling.
    participants = np.unique(dataset.presented["participant"].to_numpy())
    P = len(participants)

    boot_rng = np.random.default_rng(boot_stream_seed)
    draw_rows: list[dict[str, Any]] = []
    n_converged = 0
    for b in range(n_bootstraps):
        # Each draw's restart stream is derived deterministically from the bootstrap index.
        bootstrap_restart_seed = int(
            np.random.SeedSequence([boot_stream_seed, b]).generate_state(1)[0]
        )
        idxs = boot_rng.choice(participants, size=P, replace=True)
        try:
            resampled = _resample_participants(dataset, idxs)
            theta_b = fit_mle(
                resampled, n_restarts=n_restarts, seed=bootstrap_restart_seed,
                standard_tcm=standard_tcm,
                optional_mechanisms=optional_mechanisms,
                separate_retrieval_weights=separate_retrieval_weights,
            )
            row: dict[str, Any] = {"bootstrap_id": b, "converged": True}
            for name in free_names:
                row[name] = float(theta_b[name])
            draw_rows.append(row)
            n_converged += 1
        except FitError:
            row = {"bootstrap_id": b, "converged": False}
            for name in free_names:
                row[name] = float("nan")
            draw_rows.append(row)

    # Abort if < 90% converged.
    if n_bootstraps > 0 and n_converged / n_bootstraps < 0.9:
        raise FitError(
            f"Only {n_converged}/{n_bootstraps} bootstrap draws converged; "
            "aborting (minimum 90% required)."
        )

    # Compute CIs from the converged draws.
    lower_q = (1.0 - ci) / 2.0
    upper_q = 1.0 - lower_q
    parameters_summary: dict[str, dict[str, float]] = {}
    draws_df = pd.DataFrame(draw_rows)
    converged_mask = draws_df["converged"].to_numpy(dtype=bool) if len(draws_df) else np.array([], dtype=bool)
    for name in free_names:
        values = draws_df.loc[converged_mask, name].to_numpy() if len(draws_df) else np.array([])
        if values.size > 0:
            lo = float(np.quantile(values, lower_q))
            hi = float(np.quantile(values, upper_q))
        else:
            lo = hi = float("nan")
        parameters_summary[name] = {
            "mle": float(mle_theta[name]),
            "ci_lower": lo,
            "ci_upper": hi,
            "ci_method": "percentile_participant_bootstrap",
            "n_bootstraps": int(n_bootstraps),
            "n_converged": int(n_converged),
        }

    # Surface the derived (non-free) parameters too.
    for derived in ("w_storyline", "w_storyline_ret", "w_global_ret"):
        if derived in mle_theta and derived not in parameters_summary:
            parameters_summary[derived] = {
                "mle": float(mle_theta[derived]),
                "ci_lower": float(mle_theta[derived]),
                "ci_upper": float(mle_theta[derived]),
                "ci_method": "derived",
                "n_bootstraps": int(n_bootstraps),
                "n_converged": int(n_converged),
            }

    elapsed = perf_counter() - start_time
    manifest_bytes = str(dataset.manifest).encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    bootstrap_table = pa.Table.from_pandas(
        draws_df if len(draws_df) else pd.DataFrame(
            {"bootstrap_id": [], "converged": [],
             **{n: [] for n in free_names}}
        ),
        preserve_index=False,
    )

    return FitResult(
        parameters=parameters_summary,
        log_likelihood=float(log_like_mle),
        aic=float(aic),
        bic=float(bic),
        n_participants=int(P),
        n_lists=int(diag.n_lists),
        n_recalls_used=int(diag.n_recalls_used),
        n_intrusions_excluded=int(diag.n_intrusions_excluded),
        seed=int(seed),
        elapsed_seconds=float(elapsed),
        ms_tcm_version=str(__version__),
        dataset_manifest_sha256=manifest_hash,
        standard_tcm=bool(standard_tcm),
        bootstrap_draws=bootstrap_table,
    )
