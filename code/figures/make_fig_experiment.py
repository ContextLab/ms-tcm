"""Figure 2: FRFR-category experimental paradigm, Category condition only.

Each row of the figure shows the 16-item presentation sequence for a
sample participant-list (drawn from the bundled dataset), rendered as a
sequence of screen "cartoons" that mirror what the participant actually
saw: one word per screen, at the word's recorded screen position, in the
word's recorded display colour. A "..." marker between the first N
screens and the final recall screen indicates that some intermediate
screens are elided for space. Rows are labelled right-justified on the
left; a faint timeline bar aligns with the leftmost edge of the first
screen. No category legend (the actual stimuli are colour-varied
independently of category, as in Manning et al. 2023 Fig. 1).

Inspired by Fig. 1 of Manning et al. (2023, PsyArXiv erzfp,
https://github.com/ContextLab/FRFR-analyses).

Output: paper/figs/source/fig_experiment.pdf (+ .png preview at 96 dpi).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Which participant-list pairs to draw. Picked to illustrate the early
# (category-grouped) vs late (random) manipulation at glance. We use real
# lists from the bundled dataset so the screen positions, colours, and
# words are exactly what the participants saw.
ROWS = [
    # Participant 1 shows the category-grouped manipulation in early lists;
    # list 15 (late) is randomised. (Participant 0's lists are all randomised.)
    ("Early list (sorted)", 1, 0),
    ("Late list (random)",  1, 15),
]

# How many screens to show before the "..." marker.
N_SHOWN = 8
# Total list length (used to compute the ellipsis position).
LIST_LEN = 16


def _draw_screen(
    ax, x_left: float, y_bottom: float, w: float, h: float,
    word: str, color_rgb: tuple[float, float, float],
    pos_x: float, pos_y: float,
) -> None:
    """One FRFR screen cartoon: a rounded rectangle with a single word drawn
    at the recorded on-screen position, in the recorded display colour.
    ``pos_x, pos_y`` are in the upstream's [0, ~85] / [0, ~91] coordinate
    system (normalised screen coords). We rescale them into the box.
    """
    box = mpatches.FancyBboxPatch(
        (x_left, y_bottom), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.6, edgecolor="#444", facecolor="white",
    )
    ax.add_patch(box)
    # Rescale upstream pos to a small margin inside the box.
    inner_x = x_left + 0.12 * w + (pos_x / 85.0) * 0.76 * w
    inner_y = y_bottom + 0.18 * h + (1.0 - pos_y / 91.0) * 0.64 * h
    # Shrink the font for long words so we never overflow the box.
    fontsize = 5.0 if len(word) <= 8 else (4.2 if len(word) <= 11 else 3.5)
    ax.text(
        inner_x, inner_y, word,
        ha="center", va="center",
        fontsize=fontsize, fontweight="bold",
        color=color_rgb, family="DejaVu Sans",
    )


def _draw_recall_screen(ax, x_left: float, y_bottom: float, w: float, h: float) -> None:
    """Terminal screen: a simple microphone pictogram labelled 'free recall'."""
    box = mpatches.FancyBboxPatch(
        (x_left, y_bottom), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.6, edgecolor="#444", facecolor="#fef2f2",
    )
    ax.add_patch(box)
    cx = x_left + w / 2
    cy = y_bottom + h * 0.55
    # Microphone head (circle with a small stand).
    ax.add_patch(mpatches.Circle((cx, cy), radius=h * 0.16,
                                 facecolor="#dc2626", edgecolor="#7f1d1d",
                                 linewidth=0.6))
    # Stand/base.
    ax.plot([cx, cx], [cy - h * 0.16, cy - h * 0.30],
            color="#7f1d1d", linewidth=0.9)
    ax.plot([cx - h * 0.10, cx + h * 0.10],
            [cy - h * 0.30, cy - h * 0.30],
            color="#7f1d1d", linewidth=0.9)
    ax.text(cx, y_bottom + h * 0.12, "free recall",
            ha="center", va="center", fontsize=4.5, style="italic")


def _draw_row(
    ax, y: float, label: str,
    words: list[str], colors: list[tuple[float, float, float]],
    pos_xs: list[float], pos_ys: list[float],
    screen_w: float, screen_h: float, gap: float,
    x_start: float, label_right: float,
) -> tuple[float, float]:
    """Draw one row; return (x_left of first screen, x_right of recall screen)."""
    # Label: right-justified at label_right so its right edge sits safely to
    # the left of the first screen.
    ax.text(label_right, y + screen_h / 2, label,
            ha="right", va="center", fontsize=6.5)

    x = x_start
    for i in range(N_SHOWN):
        _draw_screen(
            ax, x, y, screen_w, screen_h,
            words[i], colors[i], pos_xs[i], pos_ys[i],
        )
        x += screen_w + gap

    # Ellipsis marker for the elided middle.
    ax.text(x + screen_w / 2, y + screen_h / 2, r"$\cdots$",
            ha="center", va="center", fontsize=9)
    x += screen_w + gap

    # One final "last screen" before recall to show we're at end-of-list.
    _draw_screen(
        ax, x, y, screen_w, screen_h,
        words[LIST_LEN - 1], colors[LIST_LEN - 1],
        pos_xs[LIST_LEN - 1], pos_ys[LIST_LEN - 1],
    )
    x += screen_w + gap

    # Recall screen.
    _draw_recall_screen(ax, x, y, screen_w, screen_h)
    x_recall_right = x + screen_w

    return x_start, x_recall_right


def draw() -> plt.Figure:
    df = pd.read_parquet("data/raw/frfr_category/presented.parquet")

    # Screen aspect 4:3 (computer-monitor-like).
    screen_h = 0.58
    screen_w = screen_h * 4.0 / 3.0
    gap = 0.08
    row_spacing = screen_h + 0.35
    # Label column: right-justified text ends a bit left of the first screen.
    label_right = 1.15
    x_start = label_right + 0.20

    # Total screens drawn per row = N_SHOWN presentation + 1 ellipsis slot +
    # 1 final presentation + 1 recall.
    n_slots = N_SHOWN + 3
    # Figure width needs to accommodate all n_slots plus right margin.
    fig_w = x_start + n_slots * (screen_w + gap) + 0.15
    fig_h = len(ROWS) * row_spacing + 0.60
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw each row and capture x extents for the timeline bar.
    first_left = None
    last_right = None
    y_top = fig_h - 0.50
    for i, (label, p, lst) in enumerate(ROWS):
        sub = (df[(df["participant"] == p) & (df["list"] == lst)]
               .sort_values("serial_position"))
        words = sub["word"].tolist()
        colors = [(r / 255.0, g / 255.0, b / 255.0)
                  for r, g, b in zip(sub["color_r"], sub["color_g"],
                                     sub["color_b"])]
        pos_xs = sub["pos_x"].tolist()
        pos_ys = sub["pos_y"].tolist()

        y = y_top - i * row_spacing
        left, right = _draw_row(
            ax, y, label, words, colors, pos_xs, pos_ys,
            screen_w, screen_h, gap, x_start, label_right,
        )
        if first_left is None:
            first_left = left
        last_right = right

    # Timeline bar: spans the full horizontal extent of the screens (starts
    # at the left edge of the leftmost screen, not inside the label gutter).
    timeline_y = y_top + screen_h + 0.18
    ax.annotate(
        "", xy=(last_right, timeline_y),
        xytext=(first_left, timeline_y),
        arrowprops=dict(arrowstyle="->", color="#555", lw=0.9),
    )
    ax.text((first_left + last_right) / 2, timeline_y + 0.10,
            "presentation phase (16 items) $\\to$ free recall",
            ha="center", va="bottom", fontsize=7)

    fig.tight_layout(pad=0.2)
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
    # Compact preview PNG for code review (dpi=96 keeps file size <200 KB).
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight",
                transparent=False, dpi=96)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
