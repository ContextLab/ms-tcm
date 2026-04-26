"""Figure 3: serial-position curve, probability of first recall, lag-CRP.

Each panel overlays observed FRFR-category data with predicted bands from
both models:

- **C&Z 2025** hierarchical free-recall (baseline): single-cue retrieval
  with hierarchical fallback to e_start. Parameters from
  ``data/processed/fits/cz_frfr/fit_summary.json`` (MLE on FRFR), or
  C&Z's published Table 1 values if no fit summary is available.
- **MS-TCM** (multi-storyline TCM, this work): C&Z + per-storyline
  contexts + λ storyline-return reinstatement + τ storyline-initiation
  mixture. Parameters from
  ``data/processed/fits/mstcm_frfr/fit_summary.json``. Overlay only
  appears if this fit summary is available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa

from ms_tcm import Dataset, load_frfr_category
from ms_tcm._likelihood_core import simulate_recalls
from ms_tcm._likelihood_core_cmr import simulate_recalls_cmr
from ms_tcm._likelihood_core_mstcm import simulate_recalls_mstcm
from ms_tcm.params import ModelParameters

# Figure scripts live under code/figures/ and import sibling helpers under
# code/analyses/. Put code/ on sys.path so the `analyses` package is
# importable without requiring code/ to be a Python package itself.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyses import lag_crp, pfr, spc as serial_position  # noqa: E402


# C&Z 2025 Table 1 published fit values (free recall on Kahana 2002).
CZ_TABLE_1 = ModelParameters(
    beta_enc=0.679,
    beta_story=0.400,    # = β_list in C&Z notation
    gamma_fc=0.315,
    k=6.50,
    beta_rec=0.326,
    epsilon_d=1.04,
    beta_rein=0.300,
    lambda_reinstate=0.0,  # MS-TCM-only; not used by C&Z hierarchical
    paradigm="free_recall",
)


def _load_fitted_params(
    fit_path: Path = Path("data/processed/fits/cz_frfr/fit_summary.json"),
) -> tuple[ModelParameters, str, float | None]:
    """Load C&Z MLE parameters from fit_summary.json if present."""
    if not fit_path.exists():
        return (
            CZ_TABLE_1,
            "C&Z 2025 (Table 1, no fit)",
            None,
        )
    summary = json.loads(fit_path.read_text())
    pp = summary["parameters"]
    params = ModelParameters(
        beta_enc=pp["beta_enc"],
        beta_story=pp["beta_list"],
        gamma_fc=pp["gamma_fc"],
        k=pp["k"],
        beta_rec=pp["beta_rec"],
        epsilon_d=pp["epsilon_d"],
        beta_rein=pp["beta_rein"],
        lambda_reinstate=0.0,
        tau_init=0.0,
        paradigm="free_recall",
    )
    metric = summary.get("log_likelihood", summary.get("rmse"))
    return (
        params,
        "C&Z 2025 (MLE)",
        float(metric) if metric is not None else None,
    )


def _load_mstcm_fitted_params(
    fit_path: Path | None = None,
) -> tuple[ModelParameters, str, float | None] | None:
    if fit_path is None:
        fit_path = Path("data/processed/fits/mstcm_frfr/fit_summary.json")
    """Load MS-TCM MLE parameters; return None if no fit summary exists."""
    if not fit_path.exists():
        return None
    summary = json.loads(fit_path.read_text())
    pp = summary["parameters"]
    # ``beta_enc_global`` and ``w_global`` are newer fields. If a stale
    # fit summary lacks them, fall back to sensible defaults.
    beta_enc_global = pp.get(
        "beta_enc_global",
        pp.get("beta_list", 0.4),
    )
    w_global = pp.get("w_global", 0.0)
    params = ModelParameters(
        beta_enc=pp["beta_enc"],
        beta_enc_global=beta_enc_global,
        beta_story=pp["beta_list"],
        gamma_fc=pp["gamma_fc"],
        k=pp["k"],
        beta_rec=pp["beta_rec"],
        epsilon_d=pp["epsilon_d"],
        beta_rein=pp["beta_rein"],
        lambda_reinstate=pp["lambda_reinstate"],
        tau_init=pp["tau_init"],
        w_global=w_global,
        paradigm="free_recall",
    )
    metric = summary.get("log_likelihood", summary.get("rmse"))
    return (
        params,
        "MS-TCM (MLE)",
        float(metric) if metric is not None else None,
    )


def _load_cmr_fitted_params(
    fit_path: Path = Path("data/processed/fits/cmr_curves_frfr/fit_summary.json"),
) -> tuple[ModelParameters, str, float | None] | None:
    """Load Polyn 2009 standard CMR MLE parameters; return None if missing."""
    if not fit_path.exists():
        return None
    summary = json.loads(fit_path.read_text())
    pp = summary["parameters"]
    params = ModelParameters(
        beta_enc=pp["beta_enc"],
        beta_story=pp.get("beta_story", pp["beta_enc"] * 0.5),  # unused
        gamma_fc=pp["gamma_fc"],
        k=pp["k"],
        beta_rec=pp["beta_rec"],
        epsilon_d=pp["epsilon_d"],
        beta_rein=pp.get("beta_rein", 0.0),
        phi_s=pp["phi_s"],
        phi_d=pp["phi_d"],
        lambda_reinstate=0.0, tau_init=0.0, w_global=0.0,
        paradigm="free_recall",
    )
    # Curve fits report rmse, not log_likelihood. Accept either key.
    metric = summary.get("log_likelihood", summary.get("rmse"))
    return (params, "Polyn 2009 CMR (MLE)", float(metric) if metric is not None else None)


def _simulate_dataset(ds: Dataset, params: ModelParameters, *,
                      n_seeds: int, master_seed: int,
                      model_kind: str = "cz") -> Dataset:
    """Sample synthetic recalls for every (participant, list) in ``ds``.

    Each (participant, list) is replicated ``n_seeds`` times with distinct
    pseudo-participant ids so the downstream curve estimators see
    N×n_seeds independent simulated lists.

    ``model_kind`` selects the simulator: ``"cz"`` (single-storyline
    hierarchical, ignores category), ``"mstcm"`` (multi-storyline with
    cat_indices from the category column), or ``"cmr"`` (Polyn 2009
    standard CMR, ignores category).
    """
    pdf = ds.presented.to_pandas()
    keys = sorted(set(zip(pdf["participant"].tolist(), pdf["list"].tolist())))
    W = ds.num_words_per_list

    pres_rows = []
    rec_rows = []
    rng_master = np.random.default_rng(master_seed)
    for seed in range(n_seeds):
        for part, lst in keys:
            pseudo_part = int(part) + 100_000 * seed
            sub_pres = pdf[
                (pdf["participant"] == part) & (pdf["list"] == lst)
            ].sort_values("serial_position").copy()
            sub_pres["participant"] = pseudo_part
            pres_rows.append(sub_pres)

            rng = np.random.default_rng(
                rng_master.integers(0, 2**31 - 1) + seed * 997
                + int(part) * 37 + int(lst),
            )

            if model_kind == "mstcm":
                # Build cat_indices from the category column.
                cat_to_idx = {}
                cat_seq = sub_pres["category"].tolist()
                for c in cat_seq:
                    if c not in cat_to_idx:
                        cat_to_idx[c] = len(cat_to_idx)
                cat_indices = np.array(
                    [cat_to_idx[c] for c in cat_seq], dtype=np.int64,
                )
                K = len(cat_to_idx)
                recalls = simulate_recalls_mstcm(
                    params, W=W, K=K,
                    cat_indices=cat_indices, rng=rng,
                )
            elif model_kind == "cmr":
                recalls = simulate_recalls_cmr(params, W=W, rng=rng)
            elif model_kind == "cz":
                recalls = simulate_recalls(params, W=W, rng=rng)
            else:
                raise ValueError(f"unknown model_kind: {model_kind!r}")

            for out_pos, sp in enumerate(recalls, start=1):
                word_row = sub_pres[sub_pres["serial_position"] == sp].iloc[0]
                rec_rows.append({
                    "participant": pseudo_part,
                    "list": int(lst),
                    "output_position": out_pos,
                    "word": str(word_row["word"]),
                    "category": str(word_row["category"]),
                    "serial_position": int(sp),
                    "list_group": str(word_row["list_group"]),
                })

    pres_df = pd.concat(pres_rows, ignore_index=True)
    rec_df = pd.DataFrame(rec_rows) if rec_rows else pd.DataFrame(
        columns=["participant", "list", "output_position", "word",
                 "category", "serial_position", "list_group"],
    )
    return Dataset(
        presented=pa.Table.from_pandas(pres_df, preserve_index=False),
        recalled=pa.Table.from_pandas(rec_df, preserve_index=False),
        manifest=ds.manifest,
    )


def _draw_band(ds: Dataset, params: ModelParameters, *,
               n_draws: int, seed: int, observe_fn,
               model_kind: str = "cz"):
    """Compute (median, lo, hi) percentile band for a per-dataset summary.

    For each of ``n_draws`` independent simulations of the full dataset,
    apply ``observe_fn`` (e.g., ``spc.observed``) and stack to a (n_draws, K)
    matrix. Return per-position (median, 5th, 95th) percentiles.
    """
    curves = []
    for d in range(n_draws):
        ds_sim = _simulate_dataset(
            ds, params, n_seeds=1, master_seed=seed + d,
            model_kind=model_kind,
        )
        curves.append(observe_fn(ds_sim))
    arr = np.stack(curves, axis=0)
    return (
        np.median(arr, axis=0),
        np.percentile(arr, 5, axis=0),
        np.percentile(arr, 95, axis=0),
    )


def _filter_to_participant(ds: Dataset, p: int) -> Dataset:
    """Return a Dataset containing only data from one participant."""
    pdf_full = ds.presented.to_pandas()
    rdf_full = ds.recalled.to_pandas()
    pdf = pdf_full[pdf_full["participant"] == p].reset_index(drop=True)
    rdf = rdf_full[rdf_full["participant"] == p].reset_index(drop=True)
    return Dataset(
        presented=pa.Table.from_pandas(pdf, preserve_index=False),
        recalled=pa.Table.from_pandas(rdf, preserve_index=False),
        manifest=ds.manifest,
    )


def _per_participant_curves(ds: Dataset, observe_fn) -> np.ndarray:
    """Apply ``observe_fn`` per participant; return (n_participants, K) array.

    Used to compute across-subjects 95% confidence intervals on observed
    behavioral curves.
    """
    pdf = ds.presented.to_pandas()
    parts = sorted(pdf["participant"].unique().tolist())
    rows = []
    for p in parts:
        ds_p = _filter_to_participant(ds, int(p))
        rows.append(observe_fn(ds_p))
    return np.stack(rows, axis=0)


def _observed_band(
    ds: Dataset, observe_fn, *, n_bootstraps: int = 2000, seed: int = 0,
):
    """Compute (mean, lo95, hi95) bootstrap CI of the across-subjects mean.

    Procedure:
      1. Compute one curve per participant (pooled across that participant's
         lists) → ``per_participant`` of shape (n_subjects, K).
      2. Bootstrap-resample participants (with replacement) ``n_bootstraps``
         times; for each resample, take the per-position MEAN across those
         resampled participants.
      3. Mean is the empirical across-participant mean of the original
         per-participant curves; 95% CI is the [2.5, 97.5] percentiles of
         the bootstrap distribution of resampled means.

    This is a standard bootstrap CI on the across-subjects mean (much
    narrower than the [2.5, 97.5] of the raw per-subject curves, which
    reflects between-subject DISPERSION, not the precision of the mean).
    """
    per_participant = _per_participant_curves(ds, observe_fn)
    n_subjects = per_participant.shape[0]
    rng = np.random.default_rng(seed)

    boot_means = np.empty((n_bootstraps, per_participant.shape[1]),
                          dtype=np.float64)
    for b in range(n_bootstraps):
        idx = rng.integers(0, n_subjects, size=n_subjects)
        boot_means[b] = np.nanmean(per_participant[idx], axis=0)

    mean = np.nanmean(per_participant, axis=0)
    lo = np.nanpercentile(boot_means, 2.5, axis=0)
    hi = np.nanpercentile(boot_means, 97.5, axis=0)
    return mean, lo, hi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-draws", type=int, default=20,
                        help="Number of independent dataset draws for band.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="paper/figs/source/fig_analyses.pdf")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--use-table-1", action="store_true",
                        help="Force using C&Z Table 1 values even if a fit "
                             "summary is available.")
    parser.add_argument("--fit-path",
                        default="data/processed/fits/cz_frfr/fit_summary.json")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = out.with_suffix(".cache.npz")
    if out.exists() and cache.exists() and not args.force_rerun:
        print(f"{out} exists; use --force-rerun to regenerate")
        return 0

    ds = load_frfr_category()
    W = ds.num_words_per_list

    if args.use_table_1:
        params, label, ll = CZ_TABLE_1, "C&Z 2025 (Table 1)", None
    else:
        # Prefer the curve-fit summary; fall back to LL fit; finally
        # to whatever path the caller passed via --fit-path.
        cz_curve_path = Path("data/processed/fits/cz_curves_frfr/fit_summary.json")
        if cz_curve_path.exists():
            params, label, ll = _load_fitted_params(cz_curve_path)
        else:
            params, label, ll = _load_fitted_params(Path(args.fit_path))

    print(f"Using parameters: {label}")
    if ll is not None:
        print(f"  fit log-likelihood: {ll:.2f}")
    for name in ("beta_enc", "beta_list", "gamma_fc", "k", "beta_rec",
                 "epsilon_d", "beta_rein"):
        val = getattr(params, name) if hasattr(params, name) else getattr(
            params, "beta_story",
        )
        print(f"  {name} = {val:.4f}")

    # Try to load MS-TCM fit too. Prefer the curve-fit summary (the
    # newer fitting objective in this work; see fit_mstcm_curves.py)
    # over the older trial-LL fit. Fall back to the LL fit if curves
    # are missing.
    mstcm_curve_path = Path("data/processed/fits/mstcm_curves_frfr/fit_summary.json")
    mstcm_ll_path = Path("data/processed/fits/mstcm_frfr/fit_summary.json")
    mstcm_loaded = (
        _load_mstcm_fitted_params(mstcm_curve_path)
        if mstcm_curve_path.exists()
        else _load_mstcm_fitted_params(mstcm_ll_path)
    )
    if mstcm_loaded is not None:
        params_mstcm, label_mstcm, ll_mstcm = mstcm_loaded
        print(f"\nUsing MS-TCM parameters: {label_mstcm}")
        if ll_mstcm is not None:
            print(f"  fit log-likelihood / rmse: {ll_mstcm:.4f}")
        for name in ("beta_enc", "beta_list", "gamma_fc", "k", "beta_rec",
                     "epsilon_d", "beta_rein", "lambda_reinstate", "tau_init"):
            val = getattr(params_mstcm, name) if hasattr(
                params_mstcm, name,
            ) else getattr(params_mstcm, "beta_story")
            print(f"  {name} = {val:.4f}")
    else:
        params_mstcm = label_mstcm = ll_mstcm = None

    # Try to load Polyn 2009 standard CMR curve fit.
    cmr_loaded = _load_cmr_fitted_params()
    if cmr_loaded is not None:
        params_cmr, label_cmr, ll_cmr = cmr_loaded
        print(f"\nUsing Polyn CMR parameters: {label_cmr}")
        if ll_cmr is not None:
            print(f"  fit metric: {ll_cmr:.4f}")
        for name in ("beta_enc", "beta_rec", "gamma_fc", "k", "epsilon_d",
                     "phi_s", "phi_d"):
            val = getattr(params_cmr, name)
            print(f"  {name} = {val:.4f}")
    else:
        params_cmr = label_cmr = ll_cmr = None

    # Split FRFR-category into early (lists 0-7) vs late (lists 8-15).
    # The two halves have systematically different category structure
    # (early lists are blocked-by-category; late lists are random) and
    # tend to produce different SPC / pFR / lag-CRP signatures.
    pdf = ds.presented.to_pandas()
    rdf = ds.recalled.to_pandas()
    halves = []
    for half_name, half_label, list_filter in [
        ("early", "Lists 1–8 (blocked by category)", lambda l: l < 8),
        ("late",  "Lists 9–16 (random order)",      lambda l: l >= 8),
    ]:
        keep_lists = sorted(set(int(l) for l in pdf["list"].unique()
                                if list_filter(int(l))))
        ds_half = Dataset(
            presented=pa.Table.from_pandas(
                pdf[pdf["list"].isin(keep_lists)].reset_index(drop=True),
                preserve_index=False,
            ),
            recalled=pa.Table.from_pandas(
                rdf[rdf["list"].isin(keep_lists)].reset_index(drop=True),
                preserve_index=False,
            ),
            manifest=ds.manifest,
        )
        halves.append((half_name, half_label, ds_half))

    rows = []
    for half_name, half_label, ds_half in halves:
        print(f"\n=== {half_label} ===")
        print(f"Computing observed measures + 95% CI ({half_name})...")
        obs = {
            "spc": _observed_band(ds_half, serial_position.observed),
            "pfr": _observed_band(ds_half, pfr.observed),
            "crp": _observed_band(ds_half, lag_crp.observed),
        }
        print(f"Drawing {args.n_draws} synthetic datasets via C&Z ({half_name})...")
        band_cz = {
            "spc": _draw_band(ds_half, params, n_draws=args.n_draws,
                              seed=args.seed,
                              observe_fn=serial_position.observed),
            "pfr": _draw_band(ds_half, params, n_draws=args.n_draws,
                              seed=args.seed, observe_fn=pfr.observed),
            "crp": _draw_band(ds_half, params, n_draws=args.n_draws,
                              seed=args.seed, observe_fn=lag_crp.observed),
        }
        if mstcm_loaded is not None:
            print(f"Drawing {args.n_draws} synthetic datasets via MS-TCM ({half_name})...")
            band_mstcm = {
                "spc": _draw_band(ds_half, params_mstcm, n_draws=args.n_draws,
                                  seed=args.seed,
                                  observe_fn=serial_position.observed,
                                  model_kind="mstcm"),
                "pfr": _draw_band(ds_half, params_mstcm, n_draws=args.n_draws,
                                  seed=args.seed, observe_fn=pfr.observed,
                                  model_kind="mstcm"),
                "crp": _draw_band(ds_half, params_mstcm, n_draws=args.n_draws,
                                  seed=args.seed, observe_fn=lag_crp.observed,
                                  model_kind="mstcm"),
            }
        else:
            band_mstcm = None
        if cmr_loaded is not None:
            print(f"Drawing {args.n_draws} synthetic datasets via Polyn CMR ({half_name})...")
            band_cmr = {
                "spc": _draw_band(ds_half, params_cmr, n_draws=args.n_draws,
                                  seed=args.seed,
                                  observe_fn=serial_position.observed,
                                  model_kind="cmr"),
                "pfr": _draw_band(ds_half, params_cmr, n_draws=args.n_draws,
                                  seed=args.seed, observe_fn=pfr.observed,
                                  model_kind="cmr"),
                "crp": _draw_band(ds_half, params_cmr, n_draws=args.n_draws,
                                  seed=args.seed, observe_fn=lag_crp.observed,
                                  model_kind="cmr"),
            }
        else:
            band_cmr = None
        rows.append({
            "half_label": half_label,
            "obs": obs,
            "cz": band_cz,
            "mstcm": band_mstcm,
            "cmr": band_cmr,
        })

    # --- Two-row figure: row 0 = early lists, row 1 = late lists ---
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=False)
    positions = np.arange(1, W + 1)
    cmr_color = "#2ca02c"       # green — Polyn 2009 standard CMR
    cz_color = "#d62728"        # red — C&Z 2025 hierarchical
    mstcm_color = "#1f77b4"     # blue — MS-TCM
    obs_color = "black"

    # Panel order across columns:
    #   col 0: pFR     — where do you start recalling?
    #   col 1: lag-CRP — how do you transition from one recall to the next?
    #   col 2: SPC     — what do you recall overall?

    lags = lag_crp.lag_axis(W)
    neg_keep = (lags >= -5) & (lags <= -1)
    pos_keep = (lags >= 1) & (lags <= 5)
    lags_neg = lags[neg_keep]
    lags_pos = lags[pos_keep]

    panel_letters = [["A", "B", "C"], ["D", "E", "F"]]

    for row_idx, row_info in enumerate(rows):
        obs = row_info["obs"]
        band_cz = row_info["cz"]
        band_mstcm = row_info["mstcm"]
        band_cmr = row_info.get("cmr")
        is_top_row = (row_idx == 0)

        # --- Column 0: pFR ---
        ax = axes[row_idx, 0]
        m_o, lo_o, hi_o = obs["pfr"]
        ax.plot(positions, m_o, "o-", color=obs_color,
                label="observed (mean ± 95% CI)",
                markersize=3.5, linewidth=1.0)
        ax.fill_between(positions, lo_o, hi_o,
                        color=obs_color, alpha=0.15, linewidth=0)
        if band_cmr is not None:
            m, lo, hi = band_cmr["pfr"]
            ax.plot(positions, m, ":", color=cmr_color,
                    label=label_cmr, linewidth=1.0)
            ax.fill_between(positions, lo, hi, color=cmr_color, alpha=0.15)
        m, lo, hi = band_cz["pfr"]
        ax.plot(positions, m, "--", color=cz_color,
                label=label, linewidth=1.0)
        ax.fill_between(positions, lo, hi, color=cz_color, alpha=0.2)
        if band_mstcm is not None:
            m, lo, hi = band_mstcm["pfr"]
            ax.plot(positions, m, "--", color=mstcm_color,
                    label=label_mstcm, linewidth=1.0)
            ax.fill_between(positions, lo, hi,
                            color=mstcm_color, alpha=0.2)
        ax.set_xlabel("serial position")
        ax.set_ylabel("P(first recall)")
        ax.set_title(
            f"{panel_letters[row_idx][0]}. {row_info['half_label']}: "
            f"probability of first recall",
            fontsize=10,
        )
        if is_top_row:
            ax.legend(fontsize=8, loc="best")

        # --- Column 1: lag-CRP ---
        ax = axes[row_idx, 1]
        m_o, lo_o, hi_o = obs["crp"]
        ax.plot(lags_neg, m_o[neg_keep], "o-", color=obs_color,
                markersize=3.5, linewidth=1.0)
        ax.fill_between(lags_neg, lo_o[neg_keep], hi_o[neg_keep],
                        color=obs_color, alpha=0.15, linewidth=0)
        ax.plot(lags_pos, m_o[pos_keep], "o-", color=obs_color,
                markersize=3.5, linewidth=1.0)
        ax.fill_between(lags_pos, lo_o[pos_keep], hi_o[pos_keep],
                        color=obs_color, alpha=0.15, linewidth=0)

        if band_cmr is not None:
            m, lo, hi = band_cmr["crp"]
            ax.plot(lags_neg, m[neg_keep], ":", color=cmr_color, linewidth=1.0)
            ax.fill_between(lags_neg, lo[neg_keep], hi[neg_keep],
                            color=cmr_color, alpha=0.15)
            ax.plot(lags_pos, m[pos_keep], ":", color=cmr_color, linewidth=1.0)
            ax.fill_between(lags_pos, lo[pos_keep], hi[pos_keep],
                            color=cmr_color, alpha=0.15)

        m, lo, hi = band_cz["crp"]
        ax.plot(lags_neg, m[neg_keep], "--", color=cz_color, linewidth=1.0)
        ax.fill_between(lags_neg, lo[neg_keep], hi[neg_keep],
                        color=cz_color, alpha=0.2)
        ax.plot(lags_pos, m[pos_keep], "--", color=cz_color, linewidth=1.0)
        ax.fill_between(lags_pos, lo[pos_keep], hi[pos_keep],
                        color=cz_color, alpha=0.2)

        if band_mstcm is not None:
            m, lo, hi = band_mstcm["crp"]
            ax.plot(lags_neg, m[neg_keep], "--", color=mstcm_color,
                    linewidth=1.0)
            ax.fill_between(lags_neg, lo[neg_keep], hi[neg_keep],
                            color=mstcm_color, alpha=0.2)
            ax.plot(lags_pos, m[pos_keep], "--", color=mstcm_color,
                    linewidth=1.0)
            ax.fill_between(lags_pos, lo[pos_keep], hi[pos_keep],
                            color=mstcm_color, alpha=0.2)

        ax.axvline(0, color="#bbb", linewidth=0.5, linestyle=":")
        ax.set_xlabel("lag")
        ax.set_ylabel("conditional response probability")
        ax.set_title(
            f"{panel_letters[row_idx][1]}. {row_info['half_label']}: "
            f"lag-CRP",
            fontsize=10,
        )

        # --- Column 2: SPC ---
        ax = axes[row_idx, 2]
        m_o, lo_o, hi_o = obs["spc"]
        ax.plot(positions, m_o, "o-", color=obs_color,
                markersize=3.5, linewidth=1.0)
        ax.fill_between(positions, lo_o, hi_o,
                        color=obs_color, alpha=0.15, linewidth=0)
        if band_cmr is not None:
            m, lo, hi = band_cmr["spc"]
            ax.plot(positions, m, ":", color=cmr_color, linewidth=1.0)
            ax.fill_between(positions, lo, hi, color=cmr_color, alpha=0.15)
        m, lo, hi = band_cz["spc"]
        ax.plot(positions, m, "--", color=cz_color, linewidth=1.0)
        ax.fill_between(positions, lo, hi, color=cz_color, alpha=0.2)
        if band_mstcm is not None:
            m, lo, hi = band_mstcm["spc"]
            ax.plot(positions, m, "--", color=mstcm_color, linewidth=1.0)
            ax.fill_between(positions, lo, hi, color=mstcm_color, alpha=0.2)
        ax.set_xlabel("serial position")
        ax.set_ylabel("P(recall)")
        ax.set_title(
            f"{panel_letters[row_idx][2]}. {row_info['half_label']}: "
            f"serial-position curve",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", transparent=True)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight",
                transparent=False, dpi=96)
    print(f"Wrote {out}")

    np.savez(cache, ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
