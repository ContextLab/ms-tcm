"""Figure: the MS-TCM v6 architecture.

v6 hierarchical CMR (two context levels: item-level + storyline-level,
associative matrices M^IC and M^SC, boundary synchronisation,
Hebbian updates) plus the single mechanism MS-TCM adds on top of
Cornell & Zhang (2025): storyline-return reinstatement at encoding
with strength lambda (v6 Eq. 6).

Python identifier convention (see notes/v6_migration.md): the math
symbol lambda is implemented as ``lambda_reinstate`` because
``lambda`` is a reserved Python keyword.

Output: paper/figs/source/fig_model.pdf (+ .png preview at dpi <= 100
per project image-resolution guidance).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _rbox(ax, xy, w, h, label, *, fc="white", ec="#222", lw=1.2, fontsize=9):
    """Rounded-rectangle label box centered at xy."""
    rect = FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, src, dst, *, label=None, color="#222", lw=1.2, rad=0.0,
           label_xy=None, fontsize=8, labelweight="normal"):
    arr = FancyArrowPatch(
        src, dst, arrowstyle="-|>", color=color,
        mutation_scale=12, linewidth=lw,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if label:
        if label_xy is None:
            label_xy = ((src[0] + dst[0]) / 2, (src[1] + dst[1]) / 2)
        ax.text(label_xy[0], label_xy[1], label, fontsize=fontsize,
                ha="center", va="center", color=color, weight=labelweight)


def make_figure(out_pdf: Path, out_png: Path, *, dpi: int = 90) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.set_axis_off()

    # --- feature input -----------------------------------------------------
    _rbox(ax, (1.1, 3.2), 1.6, 1.1,
          r"feature $f_i$" "\n" r"$\mathbf{c}^{\mathrm{IN}}_i$",
          fc="#fde68a")

    # --- item-level context (top center) ----------------------------------
    _rbox(ax, (4.6, 4.4), 2.6, 1.05,
          r"item-level $\mathbf{c}^{\mathrm{item}}$" "\n"
          r"drifts at $\beta_{\mathrm{enc}}$ every step",
          fc="#fde6c9")

    # --- storyline-level context (center) ---------------------------------
    _rbox(ax, (4.6, 2.9), 2.6, 1.05,
          r"storyline $\mathbf{c}^{\mathrm{story}}_{s(i)}$" "\n"
          r"drifts at $\beta_{\mathrm{story}}$ only while active",
          fc="#dbeafe")

    # --- M^SC cache -------------------------------------------------------
    _rbox(ax, (4.6, 1.0), 2.6, 0.85,
          r"$M^{\mathrm{SC}}$ cache" "\n"
          r"cached storyline snapshots",
          fc="#d7f0d7")

    # --- M^IC associative matrix ------------------------------------------
    _rbox(ax, (8.4, 4.4), 1.6, 1.05,
          r"$M^{\mathrm{IC}}$" "\n" r"(Hebbian)",
          fc="#fef3c7")

    # --- softmax readout --------------------------------------------------
    _rbox(ax, (8.4, 2.4), 1.6, 1.2,
          r"softmax" "\n" r"gain $k$" "\n"
          r"$P(j\mid\mathbf{c}^{\mathrm{ret}})$",
          fc="#fecaca")

    # --- arrows -----------------------------------------------------------
    _arrow(ax, (1.9, 3.5), (3.3, 4.1),
           label=r"$\beta_{\mathrm{enc}}$",
           label_xy=(2.55, 4.05))
    _arrow(ax, (1.9, 2.9), (3.3, 2.9),
           label=r"$\beta_{\mathrm{story}}$",
           label_xy=(2.55, 3.15))

    # Hebbian item->M^IC
    _arrow(ax, (5.9, 4.4), (7.6, 4.4),
           label=r"Hebb $\Delta M^{\mathrm{IC}}$",
           label_xy=(6.8, 4.65))

    # Retrieval: M^IC -> softmax
    _arrow(ax, (8.4, 3.85), (8.4, 3.05),
           label=r"$a=M^{\mathrm{IC}\!\top}\mathbf{c}^{\mathrm{ret}}$",
           label_xy=(9.5, 3.45))

    # Boundary sync (storyline -> item)
    _arrow(ax, (4.6, 3.45), (4.6, 3.85),
           label=r"boundary sync",
           label_xy=(6.05, 3.7))

    # Storyline cached to M^SC at switch
    _arrow(ax, (4.2, 2.35), (4.2, 1.45),
           label=r"switch: cache",
           label_xy=(3.2, 1.90))

    # THE NEW MECHANISM: lambda reinstatement, M^SC -> storyline
    _arrow(ax, (5.1, 1.45), (5.1, 2.35), color="#b91c1c", lw=2.0,
           label=r"$\lambda$ reinstate (v6 Eq.\ 6)",
           label_xy=(6.25, 1.85), labelweight="bold")

    # Titles
    ax.text(5.0, 5.85,
            r"MS-TCM v6 = Cornell \& Zhang (2025) hierarchical CMR $+$ storyline-return reinstatement",
            ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(5.0, 0.20,
            r"The red arrow ($\lambda$) is the sole mechanism MS-TCM adds; everything else is inherited.",
            ha="center", va="center", fontsize=9, style="italic", color="#666")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument(
        "--out-pdf", type=Path,
        default=repo_root / "paper" / "figs" / "source" / "fig_model.pdf",
    )
    parser.add_argument(
        "--out-png", type=Path,
        default=repo_root / "paper" / "figs" / "source" / "fig_model.png",
    )
    parser.add_argument("--dpi", type=int, default=90)
    parser.add_argument("--force-rerun", action="store_true",
                        help="Ignored; kept for CLI-convention compatibility.")
    args = parser.parse_args(argv)
    make_figure(args.out_pdf, args.out_png, dpi=args.dpi)
    print(f"Wrote {args.out_pdf}")
    print(f"Wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
