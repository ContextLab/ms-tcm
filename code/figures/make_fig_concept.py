"""Figure 1: the core conceptual claim of MS-TCM, as an interleaved-threads
illustration. Analogous to Polyn, Norman, & Kahana (2009) Fig. 1, but
replacing the two-spotlight metaphor with a multi-stranded braid that
captures MS-TCM's key architectural idea: storylines run on their own
internal timelines, which are spliced into the viewer's experience as
interleaved segments.

The figure has three stacked panels on a shared horizontal timeline:

  (A) The K unspooled storylines.
      Each storyline is drawn as its own horizontal strand, coloured with a
      single hue. A left-to-right lightness gradient along each strand marks
      "depth into the storyline" -- position along the strand is the
      storyline's own internal clock, independent of the viewer's timeline.

  (B) The viewer's timeline, drawn as a braid.
      A single horizontal rope made of short segments whose colours alternate
      between the K storyline hues. Each segment is a copy of the
      corresponding region of the matching strand in (A), with the SAME
      internal-clock shading: so the first "red" segment is light-red (early
      in storyline 1), a later "red" segment further down the braid is
      darker-red (later in storyline 1), even though red segments are
      separated in viewer-time by blue / green segments from other
      storylines. Thin guide arcs connect each braid segment back to the
      strand it was spliced from.

  (C) Context drifts: global + K storyline-specific.
      K+1 horizontal context rails, one above the other. The global rail
      drifts continuously (a smooth gradient that advances at every tick on
      the viewer's timeline). Each storyline rail advances only during that
      storyline's braid segments (visible as a colour gradient within those
      segments) and is flat-held everywhere else (a constant colour strip
      across the intervening segments). This makes the frozen-while-inactive
      property of MS-TCM explicit.

Output: paper/figs/source/fig_concept.pdf (+ .png preview at 96 dpi).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch


# Distinct storyline hues (colour-blind friendly where possible).
STORY_COLOURS = [
    # (light, dark) endpoints for each storyline's gradient.
    ("#fecaca", "#991b1b"),   # red / storyline A
    ("#bfdbfe", "#1e3a8a"),   # blue / storyline B
    ("#bbf7d0", "#14532d"),   # green / storyline C
]
STORY_LABELS = ["storyline A", "storyline B", "storyline C"]


def _gradient_cmap(light: str, dark: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f"g_{light}_{dark}", [light, dark], N=256,
    )


def _draw_gradient_bar(
    ax, x: float, y: float, w: float, h: float,
    cmap, t0: float = 0.0, t1: float = 1.0,
    edgecolor: str | None = "#374151", lw: float = 0.5,
) -> None:
    """Draw a horizontal rectangle filled with a linear colour gradient.
    ``t0, t1`` select the sub-range of the colormap to use (so we can
    continue a strand's gradient across disjoint segments)."""
    n = 256
    grad = np.linspace(t0, t1, n).reshape(1, -1)
    ax.imshow(
        grad, aspect="auto", cmap=cmap,
        extent=(x, x + w, y, y + h),
        zorder=2, interpolation="bilinear",
    )
    if edgecolor is not None:
        ax.add_patch(mpatches.Rectangle(
            (x, y), w, h,
            facecolor="none", edgecolor=edgecolor, linewidth=lw, zorder=3,
        ))


def _flat_bar(
    ax, x: float, y: float, w: float, h: float,
    color: str, edgecolor: str | None = "#374151", lw: float = 0.5,
) -> None:
    ax.add_patch(mpatches.Rectangle(
        (x, y), w, h,
        facecolor=color, edgecolor=edgecolor, linewidth=lw, zorder=2,
    ))


# --- The viewer's timeline: which storyline is on screen at each viewer-tick.
# 20 ticks; each tick is one event the viewer watches. The sequence of
# storyline indices below is the interleaving pattern the viewer experiences.
VIEWER_SEQUENCE = [
    0, 0, 1, 0, 1, 2, 1, 2, 0, 2, 1, 0, 2, 1, 0, 2, 0, 1, 2, 1,
]

N_STORIES = 3
N_TICKS = len(VIEWER_SEQUENCE)


def _compute_internal_clocks() -> list[list[int]]:
    """For each storyline k, return the list of viewer-ticks at which k is
    active. Position i in the returned list tells you the i-th segment of
    storyline k, i.e. the i-th step along storyline k's internal timeline."""
    clocks: list[list[int]] = [[] for _ in range(N_STORIES)]
    for t, s in enumerate(VIEWER_SEQUENCE):
        clocks[s].append(t)
    return clocks


def draw() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.0, 7.2))
    ax.set_xlim(-0.3, N_TICKS + 0.8)
    ax.set_ylim(-1.2, 11.2)
    ax.set_aspect("auto")
    ax.axis("off")

    cmaps = [_gradient_cmap(lo, hi) for lo, hi in STORY_COLOURS]
    clocks = _compute_internal_clocks()

    # ---- Panel (A): the K unspooled storylines -------------------------------
    strand_h = 0.55
    strand_gap = 0.28
    panel_a_top = 10.6
    # Leave space on the left for the storyline labels.
    label_right = 1.55
    panel_a_left = label_right + 0.12
    panel_a_right = N_TICKS + 0.4

    ax.text(-0.25, panel_a_top + 0.35,
            "A. Three storylines, each with its own internal timeline",
            ha="left", va="bottom", fontsize=10, fontweight="bold")

    strand_ys = []
    for k in range(N_STORIES):
        y = panel_a_top - (k + 1) * (strand_h + strand_gap)
        strand_ys.append(y)
        _draw_gradient_bar(
            ax, panel_a_left, y,
            panel_a_right - panel_a_left, strand_h, cmaps[k],
            t0=0.0, t1=1.0,
            edgecolor="#334155", lw=0.6,
        )
        # Faint segment dividers at each internal-clock boundary, so the
        # reader can see this strand has been "cut" into sections that the
        # braid will splice together.
        n_seg = len(clocks[k])
        for i in range(1, n_seg):
            x_div = panel_a_left + (i / n_seg) * (panel_a_right - panel_a_left)
            ax.plot([x_div, x_div], [y + 0.04, y + strand_h - 0.04],
                    color="white", lw=0.8, zorder=4, alpha=0.85)
        # Left-justified label.
        ax.text(label_right, y + strand_h / 2,
                STORY_LABELS[k],
                ha="right", va="center", fontsize=8.5)
        # "early / later" markers at left and right tips.
        ax.text(panel_a_left + 0.05, y - 0.14, "early",
                ha="left", va="top", fontsize=6.5, color="#64748b",
                style="italic")
        ax.text(panel_a_right - 0.05, y - 0.14, "later",
                ha="right", va="top", fontsize=6.5, color="#64748b",
                style="italic")

    # ---- Panel (B): the viewer's braid timeline -----------------------------
    braid_y = 5.50
    braid_h = 0.70
    braid_left = panel_a_left
    braid_right = panel_a_right
    tick_w = (braid_right - braid_left) / N_TICKS

    ax.text(-0.25, braid_y + braid_h + 0.30,
            "B. The interleaved braid the viewer experiences",
            ha="left", va="bottom", fontsize=10, fontweight="bold")

    # For each storyline we precompute where along its own internal [0, 1]
    # gradient each of its segments sits, so the braid segments use the same
    # lightness that the matching section of the unspooled strand in (A) uses.
    segment_map: dict[int, tuple[float, float]] = {}
    for k, ticks in enumerate(clocks):
        n = len(ticks)
        if n == 0:
            continue
        for i, t in enumerate(ticks):
            segment_map[t] = (i / n, (i + 1) / n)  # (t0, t1) into cmap

    for t, k in enumerate(VIEWER_SEQUENCE):
        x = braid_left + t * tick_w
        t0, t1 = segment_map[t]
        # Fill each braid segment with a SINGLE colour corresponding to its
        # position along storyline k's internal clock (midpoint of the
        # [t0, t1] sub-range). Within a segment there is no further gradient.
        colour = cmaps[k]((t0 + t1) / 2)
        _flat_bar(
            ax, x, braid_y, tick_w, braid_h,
            color=colour, edgecolor="#334155", lw=0.4,
        )
        # (No explicit strand-to-braid guide arcs: the colour + lightness of
        # each braid segment already identifies its storyline of origin and
        # its position along that storyline's internal clock. Earlier
        # iterations drew thin arcs but they cluttered the region between
        # panels A and B without adding information.)

    # Viewer-timeline axis under the braid.
    ax.annotate(
        "", xy=(braid_right + 0.05, braid_y - 0.22),
        xytext=(braid_left, braid_y - 0.22),
        arrowprops=dict(arrowstyle="-|>", color="#334155", lw=0.7),
    )
    ax.text((braid_left + braid_right) / 2, braid_y - 0.40,
            "viewer's timeline $\\to$",
            ha="center", va="top", fontsize=8, style="italic")

    # ---- Panel (C): context drifts ------------------------------------------
    ctx_top = 3.30
    ctx_h = 0.38
    ctx_gap = 0.14
    ctx_left = braid_left
    ctx_right = braid_right

    # Title well above the top rail so it never overlaps.
    ax.text(-0.25, ctx_top + ctx_h + 0.55,
            "C. Context drifts",
            ha="left", va="bottom", fontsize=10, fontweight="bold")
    ax.text(-0.25, ctx_top + ctx_h + 0.20,
            "global context drifts at every step; each storyline context drifts only during its own segments, and is frozen otherwise",
            ha="left", va="bottom", fontsize=7.5, style="italic",
            color="#475569")

    # Global context: a single rail across the full viewer timeline with a
    # continuous gradient from light to dark grey.
    global_cmap = LinearSegmentedColormap.from_list(
        "global", ["#e2e8f0", "#0f172a"], N=256,
    )
    y_g = ctx_top
    _draw_gradient_bar(
        ax, ctx_left, y_g, ctx_right - ctx_left, ctx_h, global_cmap,
        edgecolor="#334155", lw=0.6,
    )
    ax.text(ctx_left - 0.15, y_g + ctx_h / 2,
            "global $\\mathbf{c}_G$",
            ha="right", va="center", fontsize=8.5)

    # K storyline rails. For each storyline, walk along the viewer timeline
    # tick by tick. A tick where the storyline is active draws a gradient
    # segment advancing that storyline's internal clock. A tick where it is
    # inactive draws a FLAT bar at the colour the storyline had the last time
    # it was active (its "frozen" state).
    for k in range(N_STORIES):
        y_s = y_g - (k + 1) * (ctx_h + ctx_gap)
        ax.text(ctx_left - 0.15, y_s + ctx_h / 2,
                f"$\\mathbf{{c}}_{{S_{k+1}}}$ ({STORY_LABELS[k]})",
                ha="right", va="center", fontsize=8.5)
        # Find this storyline's segments.
        ticks = clocks[k]
        n = len(ticks)
        light, dark = STORY_COLOURS[k]
        cmap_k = cmaps[k]
        # Walk the viewer timeline.
        last_internal = 0.0  # fraction 0..1 of storyline-k's internal clock
        internal_step = 1.0 / max(n, 1)
        seen = 0
        for t, s in enumerate(VIEWER_SEQUENCE):
            x = ctx_left + t * tick_w
            if s == k:
                # Active: advance the internal clock by one segment and colour
                # this tick at the midpoint of the new segment. No within-
                # segment gradient; each tick is a single solid colour.
                t0 = last_internal
                t1 = last_internal + internal_step
                active_colour = cmap_k((t0 + t1) / 2)
                _flat_bar(
                    ax, x, y_s, tick_w, ctx_h,
                    color=active_colour,
                    edgecolor="#334155", lw=0.3,
                )
                last_internal = t1
                seen += 1
            else:
                # Frozen: flat hold at the last internal clock value.
                freeze_colour = cmap_k(last_internal if seen > 0 else 0.0)
                _flat_bar(
                    ax, x, y_s, tick_w, ctx_h,
                    color=freeze_colour,
                    edgecolor="#334155", lw=0.3,
                )
        # Label "frozen" on one long flat region for this storyline to drive
        # the point home. Find the longest run of inactive ticks for storyline k.
        best = (0, 0, 0)  # (length, start, end)
        run_start = None
        run_len = 0
        for t, s in enumerate(VIEWER_SEQUENCE + [k]):  # sentinel active at end
            if s != k:
                if run_start is None:
                    run_start = t
                run_len += 1
            else:
                if run_len > best[0]:
                    best = (run_len, run_start, t)
                run_start = None
                run_len = 0
        _, start, end = best
        if best[0] >= 3:
            mid_x = ctx_left + ((start + end) / 2) * tick_w
            ax.text(mid_x, y_s - 0.16, "frozen",
                    ha="center", va="top", fontsize=6.5,
                    style="italic", color="#475569")

    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="paper/figs/source/fig_concept.pdf")
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
                transparent=False, dpi=96)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
