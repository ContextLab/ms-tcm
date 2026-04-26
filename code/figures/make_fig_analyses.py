"""Figure 3: serial-position curve, probability of first recall, lag-CRP.

Each panel overlays observed FRFR-category data with predicted bands from
the Cornell & Zhang 2025 hierarchical free-recall model. The model is
parameterized by an MLE fit to FRFR-category if available
(``data/processed/fits/cz_frfr/fit_summary.json``); otherwise falls back
to C&Z's published Table 1 values.

Inspired by Manning et al. 2023 FRFR Fig. 3 and H&K 2002 Fig. 1.
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
    """Load MLE parameters from fit_summary.json if present.

    Returns ``(params, label, log_likelihood_or_None)``. Falls back to
    C&Z Table 1 values when the fit summary is missing.
    """
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
        beta_story=pp["beta_list"],   # = beta_story field in our schema
        gamma_fc=pp["gamma_fc"],
        k=pp["k"],
        beta_rec=pp["beta_rec"],
        epsilon_d=pp["epsilon_d"],
        beta_rein=pp["beta_rein"],
        lambda_reinstate=0.0,
        paradigm="free_recall",
    )
    return (
        params,
        "C&Z 2025 (MLE)",
        float(summary["log_likelihood"]),
    )


def _simulate_dataset(ds: Dataset, params: ModelParameters, *,
                      n_seeds: int, master_seed: int) -> Dataset:
    """Sample synthetic recalls for every (participant, list) in ``ds``.

    Each (participant, list) is replicated ``n_seeds`` times with distinct
    pseudo-participant ids so the downstream curve estimators see N×n_seeds
    independent simulated lists.

    The C&Z hierarchical free-recall sampler treats each list as a single
    storyline (we ignore FRFR-category's category structure for this
    figure — that's a deliberate simplification: C&Z's model has no
    storyline-level mechanism, so we apply its single-list free-recall
    behavior to each FRFR list independently).
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
            recalls = simulate_recalls(params, W=W, rng=rng)
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
               n_draws: int, seed: int, observe_fn):
    """Compute (median, lo, hi) percentile band for a per-dataset summary.

    For each of ``n_draws`` independent simulations of the full dataset,
    apply ``observe_fn`` (e.g., ``spc.observed``) and stack to a (n_draws, K)
    matrix. Return per-position (median, 5th, 95th) percentiles.
    """
    curves = []
    for d in range(n_draws):
        ds_sim = _simulate_dataset(ds, params, n_seeds=1, master_seed=seed + d)
        curves.append(observe_fn(ds_sim))
    arr = np.stack(curves, axis=0)
    return (
        np.median(arr, axis=0),
        np.percentile(arr, 5, axis=0),
        np.percentile(arr, 95, axis=0),
    )


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

    print("Computing observed measures (FRFR-category)...")
    obs_spc = serial_position.observed(ds)
    obs_pfr = pfr.observed(ds)
    obs_crp = lag_crp.observed(ds)

    print(f"Drawing {args.n_draws} synthetic datasets...")
    band_spc = _draw_band(
        ds, params, n_draws=args.n_draws, seed=args.seed,
        observe_fn=serial_position.observed,
    )
    band_pfr = _draw_band(
        ds, params, n_draws=args.n_draws, seed=args.seed,
        observe_fn=pfr.observed,
    )
    band_crp = _draw_band(
        ds, params, n_draws=args.n_draws, seed=args.seed,
        observe_fn=lag_crp.observed,
    )

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    positions = np.arange(1, W + 1)
    cz_color = "#d62728"

    # Panel A: serial-position curve.
    ax = axes[0]
    ax.plot(positions, obs_spc, "o-", color="black",
            label="observed", markersize=3.5, linewidth=1.0)
    m, lo, hi = band_spc
    ax.plot(positions, m, "--", color=cz_color,
            label=label, linewidth=1.0)
    ax.fill_between(positions, lo, hi, color=cz_color, alpha=0.2)
    ax.set_xlabel("serial position")
    ax.set_ylabel("P(recall)")
    ax.set_title("A. Serial-position curve")
    ax.legend(fontsize=8, loc="best")

    # Panel B: PFR.
    ax = axes[1]
    ax.plot(positions, obs_pfr, "o-", color="black",
            label="observed", markersize=3.5, linewidth=1.0)
    m, lo, hi = band_pfr
    ax.plot(positions, m, "--", color=cz_color,
            label=label, linewidth=1.0)
    ax.fill_between(positions, lo, hi, color=cz_color, alpha=0.2)
    ax.set_xlabel("serial position")
    ax.set_ylabel("P(first recall)")
    ax.set_title("B. Probability of first recall")
    ax.legend(fontsize=8, loc="best")

    # Panel C: lag-CRP.
    ax = axes[2]
    lags = lag_crp.lag_axis(W)
    keep = (np.abs(lags) >= 1) & (np.abs(lags) <= 5)
    ax.plot(lags[keep], obs_crp[keep], "o-", color="black",
            label="observed", markersize=3.5, linewidth=1.0)
    m, lo, hi = band_crp
    ax.plot(lags[keep], m[keep], "--", color=cz_color,
            label=label, linewidth=1.0)
    ax.fill_between(lags[keep], lo[keep], hi[keep],
                    color=cz_color, alpha=0.2)
    ax.axhline(0, color="#bbb", linewidth=0.5)
    ax.set_xlabel("lag")
    ax.set_ylabel("conditional response probability")
    ax.set_title("C. Lag-CRP")
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", transparent=True)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight",
                transparent=False, dpi=96)
    print(f"Wrote {out}")

    np.savez(cache, ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
