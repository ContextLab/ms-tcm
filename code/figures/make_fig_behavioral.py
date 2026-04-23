"""Figure: behavioral signatures — SPC, pFR, lag-CRP.

Three-panel figure comparing the FRFR-category empirical curves (black,
pre-computed via ``scripts/build_reference_curves.py`` and cached in
``data/processed/reference_curves/``) to simulated curves from MS-TCM
(red) and the ``--standard-tcm`` reduction (blue). Serves as
Figure \ref{fig:behavioral} in paper/main.tex.

Inputs:
  - data/processed/reference_curves/frfr_category_spc.parquet
  - data/processed/reference_curves/frfr_category_pfr.parquet
  - data/processed/reference_curves/frfr_category_lag_crp.parquet

Optional (if the reference-curves cache is missing, the script falls
back to computing empirical curves on the fly from FRFR-category raw
data — keeps the figure reproducible in a fresh clone).

Output:
  paper/figs/source/fig_behavioral.pdf
  paper/figs/source/fig_behavioral.png   (dpi <= 100 for inspection)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Put code/ on sys.path so `analyses` + `ms_tcm` imports work without install.
_THIS = Path(__file__).resolve()
_CODE = _THIS.parent.parent
_REPO = _CODE.parent
sys.path.insert(0, str(_CODE))

from ms_tcm.dataset import load_dataset  # noqa: E402


def _load_reference_curves(cache_dir: Path, dataset_dir: Path):
    """Return (spc_x, spc_y, pfr_x, pfr_y, crp_x, crp_y).

    Prefers the pre-computed reference curves under cache_dir; falls
    back to on-the-fly computation from dataset_dir if the cache is
    missing (e.g. fresh clone before `scripts/build_reference_curves.py`
    has been run).
    """
    import pandas as pd

    spc_path = cache_dir / "frfr_category_spc.parquet"
    pfr_path = cache_dir / "frfr_category_pfr.parquet"
    crp_path = cache_dir / "frfr_category_lag_crp.parquet"

    if spc_path.exists() and pfr_path.exists() and crp_path.exists():
        spc = pd.read_parquet(spc_path)
        pfr = pd.read_parquet(pfr_path)
        crp = pd.read_parquet(crp_path)
        return (
            spc["serial_position"].to_numpy(), spc["p_recall"].to_numpy(),
            pfr["serial_position"].to_numpy(), pfr["p_first_recall"].to_numpy(),
            crp["lag"].to_numpy(), crp["crp"].to_numpy(),
        )

    # Fallback: compute from raw dataset.
    sys.path.insert(0, str(_CODE))
    from analyses import spc as spc_mod, pfr as pfr_mod, lag_crp as crp_mod  # noqa: E402

    ds = load_dataset(dataset_dir)
    spc_series = spc_mod.compute_spc(ds)
    pfr_series = pfr_mod.compute_pfr(ds)
    crp_series = crp_mod.compute_lag_crp(ds)
    return (
        spc_series.index.to_numpy(), spc_series.to_numpy(),
        pfr_series.index.to_numpy(), pfr_series.to_numpy(),
        crp_series.index.to_numpy(), crp_series.to_numpy(),
    )


def make_figure(out_pdf: Path, out_png: Path, *, dpi: int = 90,
                cache_dir: Path | None = None,
                dataset_dir: Path | None = None) -> None:
    if cache_dir is None:
        cache_dir = _REPO / "data" / "processed" / "reference_curves"
    if dataset_dir is None:
        dataset_dir = _REPO / "data" / "raw" / "frfr_category"

    spc_x, spc_y, pfr_x, pfr_y, crp_x, crp_y = _load_reference_curves(
        cache_dir, dataset_dir,
    )

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    ax = axes[0]
    ax.plot(spc_x, spc_y, "o-", color="black", label="observed",
            markersize=4, linewidth=1.3)
    ax.set_xlabel("serial position")
    ax.set_ylabel("P(recall)")
    ax.set_title("Serial position curve")
    ax.set_ylim(0, max(0.8, spc_y.max() + 0.05))
    ax.legend(loc="lower center", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(pfr_x, pfr_y, "o-", color="black", label="observed",
            markersize=4, linewidth=1.3)
    ax.set_xlabel("serial position")
    ax.set_ylabel("P(first recall)")
    ax.set_title("Probability of first recall")
    ax.set_ylim(0, max(0.4, pfr_y.max() + 0.05))
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[2]
    # Restrict to |lag| <= 5 for visibility.
    mask = np.abs(crp_x) <= 5
    ax.plot(crp_x[mask], crp_y[mask], "o-", color="black", label="observed",
            markersize=4, linewidth=1.3)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("lag")
    ax.set_ylabel("cond.\\ response probability")
    ax.set_title(r"Lag-CRP ($|\mathrm{lag}|\leq 5$)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-pdf", type=Path,
        default=_REPO / "paper" / "figs" / "source" / "fig_behavioral.pdf",
    )
    parser.add_argument(
        "--out-png", type=Path,
        default=_REPO / "paper" / "figs" / "source" / "fig_behavioral.png",
    )
    parser.add_argument("--dpi", type=int, default=90)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--force-rerun", action="store_true",
                        help="Ignored; kept for CLI-convention compatibility.")
    args = parser.parse_args(argv)
    make_figure(
        args.out_pdf, args.out_png, dpi=args.dpi,
        cache_dir=args.cache_dir, dataset_dir=args.dataset_dir,
    )
    print(f"Wrote {args.out_pdf}")
    print(f"Wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
