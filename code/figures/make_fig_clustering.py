"""Figure 4: temporal, category, and semantic clustering scores with observed
vs predicted bars. Percentile-rank clustering scores per Polyn et al. 2009.
Semantic distances use sentence-transformer embeddings (default MiniLM).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ms_tcm import load_frfr_category

# Figure scripts live under code/figures/ and import sibling helpers under
# code/analyses/. Put code/ on sys.path so the `analyses` package is
# importable without requiring code/ to be a Python package itself.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyses import clustering  # noqa: E402
from analyses.embeddings import load_embeddings  # noqa: E402
from analyses.predicted import (  # noqa: E402
    compute_scalar_band, draw_synthetic_datasets, params_from_fit_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", default="data/processed/fits/mstcm")
    parser.add_argument("--n-draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embeddings",
                        default="data/processed/embeddings/frfr_category.parquet")
    parser.add_argument("--out", default="paper/figs/source/fig_clustering.pdf")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force_rerun:
        print(f"{out} exists; use --force-rerun to regenerate")
        return 0

    ds = load_frfr_category()
    obs_temporal = clustering.observed_temporal(ds)
    obs_category = clustering.observed_category(ds)

    emb_path = Path(args.embeddings)
    embeddings = None
    obs_semantic = float("nan")
    if emb_path.exists():
        embeddings = load_embeddings(str(emb_path))
        obs_semantic = clustering.observed_semantic(ds, embeddings)
    else:
        print(f"Embeddings not found at {emb_path}; semantic bar will be NaN.")

    pred_temporal = pred_category = pred_semantic = None
    if Path(args.fit, "fit_summary.json").exists():
        params = params_from_fit_summary(args.fit)
        synths = draw_synthetic_datasets(ds, params, args.n_draws, args.seed)
        pred_temporal = compute_scalar_band(synths, clustering.observed_temporal)
        pred_category = compute_scalar_band(synths, clustering.observed_category)
        if embeddings is not None:
            pred_semantic = compute_scalar_band(
                synths, lambda d: clustering.observed_semantic(d, embeddings),
            )

    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    labels = ["Temporal", "Category", "Semantic"]
    obs_vals = [obs_temporal, obs_category, obs_semantic]
    x_obs = np.arange(len(labels)) - 0.18
    x_pred = np.arange(len(labels)) + 0.18
    ax.bar(x_obs, obs_vals, width=0.32, color="black", label="observed")
    preds = [pred_temporal, pred_category, pred_semantic]
    if any(p is not None for p in preds):
        means = [p[0] if p else np.nan for p in preds]
        lo = [p[1] if p else np.nan for p in preds]
        hi = [p[2] if p else np.nan for p in preds]
        yerr = np.vstack([
            [m - l if (m == m and l == l) else 0 for m, l in zip(means, lo)],
            [h - m if (m == m and h == h) else 0 for m, h in zip(means, hi)],
        ])
        ax.bar(x_pred, means, width=0.32, color="#d62728",
               label="MS-TCM", yerr=yerr, capsize=3)

    ax.axhline(0.5, color="#bbb", linewidth=0.6, linestyle=":")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("clustering score (percentile rank)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Clustering by temporal, category, and semantic dimensions")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", transparent=True)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight",
                transparent=False, dpi=200)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
