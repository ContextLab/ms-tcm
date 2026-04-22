"""Figure 1: schematic of the Multi-Stream Temporal Context Model.

Inspired by Polyn et al. 2009 Fig. 2 (CMR) and Howard & Kahana 2002 Fig. 4
(TCM). Shows the feature layer F, the global context c_G, K storyline
contexts c_{S_1}..c_{S_K}, the composite encoding/recall cue context, and
the associative matrices M^{FC} / M^{CF}.

Output: paper/figs/source/fig_model.pdf (+ .png preview).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COL_F = "#fef3c7"
COL_CTX = "#e0f2fe"
COL_STORY = "#fce7f3"
COL_COMP = "#ecfdf5"
COL_MAT = "#dbeafe"


def _box(ax, x, y, w, h, label, color=COL_CTX, fontsize=9):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06",
        linewidth=0.9, edgecolor="black", facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize)


def _arrow(ax, x1, y1, x2, y2, label=None, dx=0.0, dy=0.15, fontsize=8):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
        linewidth=0.9, color="black",
    )
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, label,
                ha="center", va="center", fontsize=fontsize, style="italic")


def _panel_encoding(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(5, 7.7, "A. Encoding", ha="center", va="center",
            fontsize=11, fontweight="bold")
    _box(ax, 3.0, 0.5, 4.0, 1.0,
         r"feature layer $F$" + "\n" +
         r"(multi-hot: category, size, letter, $\ldots$)",
         color=COL_F, fontsize=9)
    _box(ax, 3.0, 2.4, 4.0, 0.7, r"$\mathbf{M}^{FC}$: $F \to C$",
         color=COL_MAT, fontsize=8.5)
    _box(ax, 1.1, 4.0, 3.4, 1.3,
         r"global context $\mathbf{c}_G(t)$" + "\n" +
         r"drifts at every step",
         color=COL_CTX, fontsize=8.5)
    _box(ax, 5.5, 4.0, 3.4, 1.3,
         r"storyline contexts" + "\n" +
         r"$\mathbf{c}_{S_1}(t), \ldots, \mathbf{c}_{S_K}(t)$" + "\n" +
         r"only active $S^{*}$ drifts",
         color=COL_STORY, fontsize=8.5)
    _box(ax, 2.4, 6.2, 5.2, 1.0,
         r"$\mathbf{c}_{\mathrm{comp}}(i) = w_G\,\mathbf{c}_G(t)"
         r" + w_S\,\mathbf{c}_{S^{*}}(t)$",
         color=COL_COMP, fontsize=9.5)
    _arrow(ax, 5.0, 1.55, 5.0, 2.4, label=r"$\mathbf{c}^{IN}_i$", dx=0.4, dy=0.0)
    _arrow(ax, 4.2, 3.1, 3.2, 4.0, label=r"$\beta_G$", dx=-0.35, dy=-0.05)
    _arrow(ax, 5.8, 3.1, 6.8, 4.0, label=r"$\beta_S$ (if $S^{*}$)",
           dx=0.65, dy=-0.05)
    _arrow(ax, 3.2, 5.3, 4.5, 6.2, label=r"$w_G$", dx=-0.3, dy=0.0)
    _arrow(ax, 6.8, 5.3, 5.5, 6.2, label=r"$w_S$", dx=0.3, dy=0.0)


def _panel_recall_cue(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(5, 7.7, "B. Recall cueing",
            ha="center", va="center", fontsize=11, fontweight="bold")
    # Retrieval composite cue at top.
    _box(ax, 2.2, 6.1, 5.6, 1.0,
         r"$\mathbf{c}_{\mathrm{ret}}(j) = w^{\mathrm{ret}}_G\,\mathbf{c}_G(t_j)"
         r" + w^{\mathrm{ret}}_S\,\mathbf{c}_{S_j}(t_j)$",
         color=COL_COMP, fontsize=9.5)
    # Decision rule below the cue. Split across two lines so the fraction is
    # legible in mathtext (matplotlib's parser rejects \displaystyle/\bigl).
    _box(ax, 1.4, 3.9, 7.2, 1.6,
         r"$s_i = \mathrm{sim}(\mathbf{c}_{\mathrm{ret}}(j),\,"
         r"\mathbf{c}_{\mathrm{comp}}(i)) \cdot a_i \cdot "
         r"e^{-\lambda I_{ij}}$" + "\n" +
         r"$P(i \mid j) = \exp(s_i) / \sum_{k=1}^{W} \exp(s_k)$",
         color=COL_COMP, fontsize=9)
    # Candidate set at the bottom.
    _box(ax, 1.4, 1.2, 7.2, 1.6,
         r"candidate set: every presented word $\{i : 1 \leq i \leq W\}$" + "\n" +
         r"with activation bias $a_i$ and interference $I_{ij}$ ($\S$5.3)",
         color=COL_F, fontsize=9)
    # Arrows (labels offset to avoid the arrow lines).
    _arrow(ax, 5.0, 6.1, 5.0, 5.5, dy=0.0)
    _arrow(ax, 5.0, 3.9, 5.0, 2.8, dy=0.0,
           label=r"evaluate for every candidate",
           dx=2.1, fontsize=7.5)


def draw() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    _panel_encoding(axes[0])
    _panel_recall_cue(axes[1])
    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="paper/figs/source/fig_model.pdf",
    )
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force_rerun:
        print(f"{out} exists; use --force-rerun to regenerate")
        return 0

    fig = draw()
    fig.savefig(out, bbox_inches="tight", transparent=True)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight",
                transparent=False, dpi=200)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
