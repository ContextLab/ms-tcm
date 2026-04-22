"""Figure 2: the FRFR-category experimental setup.

One row per condition (early list = sorted by category; late list = random).
Each row shows 16 word cells colored by category, plus a microphone icon on
the right to indicate the verbal recall phase. Adapted from Manning et al.
(2023) FRFR Fig. 1 (but trimmed to only the category condition since that's
all our paper uses).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


CATEGORIES = ["MAMMALS", "FRUITS", "BUILDING RELATED", "KITCHEN-RELATED"]
CATEGORY_COLORS = {
    "MAMMALS": "#f97316",
    "FRUITS": "#22c55e",
    "BUILDING RELATED": "#3b82f6",
    "KITCHEN-RELATED": "#a855f7",
}


def _draw_row(ax, y: float, list_label: str, order: list[str], words: list[str]):
    ax.text(-0.3, y + 0.35, list_label, ha="right", va="center", fontsize=9)
    for i, (cat, w) in enumerate(zip(order, words)):
        rect = mpatches.Rectangle(
            (i * 0.9, y), 0.85, 0.7,
            linewidth=0.6, edgecolor="black",
            facecolor=CATEGORY_COLORS[cat], alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(i * 0.9 + 0.425, y + 0.35, w,
                ha="center", va="center", fontsize=6.5, color="white",
                fontweight="bold")
    # A small recall-phase indicator (the FRFR task asks for free recall at
    # the end of every list; we draw it as an arrow into a labeled box).
    x_arrow_from = len(order) * 0.9 + 0.05
    x_arrow_to = x_arrow_from + 0.55
    ax.annotate(
        "", xy=(x_arrow_to, y + 0.35), xytext=(x_arrow_from, y + 0.35),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )
    recall_box = mpatches.FancyBboxPatch(
        (x_arrow_to, y), 1.4, 0.7,
        boxstyle="round,pad=0.05",
        linewidth=0.6, edgecolor="black", facecolor="#f3f4f6",
    )
    ax.add_patch(recall_box)
    ax.text(x_arrow_to + 0.7, y + 0.35, "free recall",
            ha="center", va="center", fontsize=7.5, style="italic")


def draw() -> plt.Figure:
    rng = np.random.default_rng(0)
    words_by_cat = {
        "MAMMALS": ["HORSE", "CAT", "DOG", "MOUSE"],
        "FRUITS": ["APPLE", "KIWI", "MANGO", "GRAPE"],
        "BUILDING RELATED": ["WINDOW", "DOOR", "HALL", "WALL"],
        "KITCHEN-RELATED": ["SPATULA", "GLASS", "KNIFE", "PLATE"],
    }
    # Early list: grouped by category (all of one cat, then all of next...).
    early_order: list[str] = []
    early_words: list[str] = []
    for cat in CATEGORIES:
        for w in words_by_cat[cat]:
            early_order.append(cat)
            early_words.append(w)
    # Late list: random.
    pairs = list(zip(early_order, early_words))
    rng.shuffle(pairs)
    late_order = [p[0] for p in pairs]
    late_words = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    _draw_row(ax, 1.2, r"Early list (sorted)", early_order, early_words)
    _draw_row(ax, 0.2, r"Late list (random)", late_order, late_words)

    ax.set_xlim(-2.5, 17.3)
    ax.set_ylim(-0.3, 2.3)
    ax.axis("off")

    # Legend.
    handles = [
        mpatches.Patch(color=CATEGORY_COLORS[c], label=c.title()) for c in CATEGORIES
    ]
    ax.legend(handles=handles, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8)

    ax.text(6.5, 2.1,
            "Feature-rich free recall: 30 participants, 16 lists/pt, 16 words/list, 4 categories/list",
            ha="center", va="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="paper/figs/source/fig_experiment.pdf")
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
