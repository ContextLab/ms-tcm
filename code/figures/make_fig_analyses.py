"""Figure 3: serial-position curve, probability of first recall, and lag-CRP.
Each panel overlays observed FRFR-category data with MS-TCM and (optionally)
standard-TCM predicted bands.

Inspired by Manning et al. 2023 FRFR Fig. 3 and H&K 2002 Fig. 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ms_tcm import Dataset, load_frfr_category
from code.analyses import lag_crp, pfr, serial_position
from code.analyses.predicted import (
    compute_band, draw_synthetic_datasets, params_from_fit_summary,
)


def _overlay_band(ax, x, observed_y, band, label, color):
    median, lo, hi = band
    ax.plot(x, observed_y, "o-", color=color, label=f"{label}: observed",
            markersize=4, linewidth=1.2)
    ax.plot(x, median, "--", color=color, alpha=0.8,
            label=f"{label}: MS-TCM", linewidth=1.0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", default="data/processed/fits/mstcm")
    parser.add_argument("--tcm-fit", default="data/processed/fits/tcm",
                        help="Optional standard-TCM fit; skipped if missing.")
    parser.add_argument("--n-draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="paper/figs/source/fig_analyses.pdf")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = out.with_suffix(".cache.npz")
    if out.exists() and cache.exists() and not args.force_rerun:
        print(f"{out} exists; use --force-rerun to regenerate")
        return 0

    ds = load_frfr_category()
    W = ds.num_words_per_list

    print("Computing observed measures...")
    obs_spc = serial_position.observed(ds)
    obs_pfr = pfr.observed(ds)
    obs_crp = lag_crp.observed(ds)

    mstcm_bands = None
    if Path(args.fit, "fit_summary.json").exists():
        print(f"Loading MS-TCM fit from {args.fit}...")
        params = params_from_fit_summary(args.fit)
        print(f"Drawing {args.n_draws} synthetic datasets...")
        synths = draw_synthetic_datasets(ds, params, args.n_draws, args.seed)
        mstcm_bands = {
            "spc": compute_band(synths, serial_position.observed),
            "pfr": compute_band(synths, pfr.observed),
            "crp": compute_band(synths, lag_crp.observed),
        }

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    # Panel A: serial position curve.
    ax = axes[0]
    positions = np.arange(1, W + 1)
    ax.plot(positions, obs_spc, "o-", color="black",
            label="observed", markersize=3.5, linewidth=1.0)
    if mstcm_bands:
        m, lo, hi = mstcm_bands["spc"]
        ax.plot(positions, m, "--", color="#d62728",
                label="MS-TCM", linewidth=1.0)
        ax.fill_between(positions, lo, hi, color="#d62728", alpha=0.2)
    ax.set_xlabel("serial position")
    ax.set_ylabel("P(recall)")
    ax.set_title("A. Serial-position curve")
    ax.legend(fontsize=8, loc="best")

    # Panel B: PFR.
    ax = axes[1]
    ax.plot(positions, obs_pfr, "o-", color="black",
            label="observed", markersize=3.5, linewidth=1.0)
    if mstcm_bands:
        m, lo, hi = mstcm_bands["pfr"]
        ax.plot(positions, m, "--", color="#d62728",
                label="MS-TCM", linewidth=1.0)
        ax.fill_between(positions, lo, hi, color="#d62728", alpha=0.2)
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
    if mstcm_bands:
        m, lo, hi = mstcm_bands["crp"]
        ax.plot(lags[keep], m[keep], "--", color="#d62728",
                label="MS-TCM", linewidth=1.0)
        ax.fill_between(lags[keep], lo[keep], hi[keep],
                        color="#d62728", alpha=0.2)
    ax.axhline(0, color="#bbb", linewidth=0.5)
    ax.set_xlabel("lag")
    ax.set_ylabel("conditional response probability")
    ax.set_title("C. Lag-CRP")
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", transparent=True)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight",
                transparent=False, dpi=200)
    print(f"Wrote {out}")

    # Tiny cache marker so idempotent check works.
    np.savez(cache, ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
