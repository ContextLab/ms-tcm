"""Figure 2: the Multi-Stream Temporal Context Model architecture.

Analogous in purpose to Fig. 2 of Polyn, Norman, & Kahana (2009), which
shows CMR's F/C layers, associative matrices, and decision competition.
MS-TCM is a different animal -- there is no single context layer with
sub-regions, and no learned F->C / C->F associative matrices. Instead
the model maintains K+1 context vectors in parallel (one global + one per
storyline) with different update rules, and a mixing stage that builds
the encoding and retrieval composites from those context vectors. This
figure makes that architecture explicit.

The figure is organised top-to-bottom as a single data-flow:

  (1) Environment -> event -> feature input  c^IN_i
  (2) Feature input drives two parallel kinds of context store:
      c_G drifts every step; c_{S*} drifts only when its storyline is
      active; the inactive c_{S_k} are frozen.
  (3) At encoding, c_G and c_{S*} are mixed via w_G, w_S into the
      composite c_comp(i). At retrieval, the cue event j's contexts are
      mixed via (possibly different) w_G^ret, w_S^ret into c_ret(j).
  (4) The decision stage scores every candidate i's c_comp(i) against
      c_ret(j) via cosine similarity, optionally down-weighted by
      exp(-lambda I_{ij}), and softmaxes into P(i | j).

Output: paper/figs/source/fig_model.pdf (+ .png preview at 96 dpi).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# Palette shared with fig_concept.
COL_EVENT   = "#fde68a"
COL_GLOBAL  = "#dbeafe"
COL_STORY   = "#fce7f3"
COL_STORY_A = "#fb7185"
COL_COMP    = "#d1fae5"
COL_DECIDE  = "#fef3c7"


def _rbox(ax, x, y, w, h, facecolor="white", edgecolor="#222", lw=0.8, **kw):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.10",
        linewidth=lw, edgecolor=edgecolor, facecolor=facecolor, **kw,
    ))


def _arrow(ax, x1, y1, x2, y2, rad=0.0, color="#222", lw=1.0,
           label=None, label_offset=(0.0, 0.18), label_size=8,
           label_anchor="mid"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}",
    ))
    if label is not None:
        if label_anchor == "start":
            mx, my = x1, y1
        elif label_anchor == "end":
            mx, my = x2, y2
        else:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label,
                ha="center", va="center", fontsize=label_size,
                style="italic")


def _vector_bar(ax, x, y, w, h, label, colour, n_cells=8, highlight=False):
    """Draw a context vector as a row of tiny cells inside a rounded box."""
    _rbox(
        ax, x, y, w, h, facecolor=colour,
        edgecolor=("#be185d" if highlight else "#334155"),
        lw=(1.1 if highlight else 0.7),
    )
    pad = 0.10
    cw = (w - 2 * pad) / n_cells
    ch = h * 0.40
    cy = y + (h - ch) / 2 - 0.05
    for i in range(n_cells):
        ax.add_patch(mpatches.Rectangle(
            (x + pad + i * cw + 0.012, cy), cw - 0.024, ch,
            facecolor="white", edgecolor="#64748b", lw=0.35,
        ))
    ax.text(x + w / 2, y + h - 0.15, label,
            ha="center", va="top", fontsize=8)


def draw() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.5, 11.0)
    ax.axis("off")

    # =====================================================================
    # (1) Environment -> event -> feature input
    # =====================================================================
    env_y = 9.80
    ax.text(0.2, env_y, "the environment",
            ha="left", va="center", fontsize=9.5, style="italic",
            color="#475569")

    ev_x, ev_y, ev_w, ev_h = 6.4, env_y - 0.55, 3.2, 1.10
    _rbox(ax, ev_x, ev_y, ev_w, ev_h, facecolor=COL_EVENT,
          edgecolor="#78350f")
    ax.text(ev_x + ev_w / 2, ev_y + ev_h / 2,
            "event $i$ at time $t$\n(on storyline $S^{*}$)",
            ha="center", va="center", fontsize=9.5)

    # Environment arrow -> event.
    _arrow(ax, 2.2, env_y, ev_x - 0.05, env_y, lw=0.9)

    # Feature-input label below event (with arrow down into the stores band).
    cin_x = ev_x + ev_w / 2
    cin_y = ev_y - 0.45
    ax.text(cin_x, cin_y,
            "feature input $\\mathbf{c}^{\\mathrm{IN}}_i$",
            ha="center", va="center", fontsize=9, fontweight="bold")
    _arrow(ax, cin_x, ev_y - 0.05, cin_x, cin_y + 0.20, lw=0.9)

    # =====================================================================
    # (2) K+1 context stores
    # =====================================================================
    ctx_y = 6.30
    ctx_h = 0.90

    cg_x, cg_w = 1.0, 3.4
    _vector_bar(
        ax, cg_x, ctx_y, cg_w, ctx_h,
        "global $\\mathbf{c}_G(t)$", COL_GLOBAL,
    )
    ax.text(cg_x + cg_w / 2, ctx_y - 0.22,
            "drifts at every step",
            ha="center", va="top", fontsize=7.5,
            style="italic", color="#475569")

    # Three storyline stores on the right; S_2 is active.
    cs_left = 5.60
    cs_gap = 0.20
    cs_w = 3.00
    active_k = 1
    cs_xs = []
    for k in range(3):
        x = cs_left + k * (cs_w + cs_gap)
        cs_xs.append(x)
        _vector_bar(
            ax, x, ctx_y, cs_w, ctx_h,
            f"$\\mathbf{{c}}_{{S_{k+1}}}(t)$",
            COL_STORY_A if k == active_k else COL_STORY,
            highlight=(k == active_k),
        )
    # Caption spanning the storyline stores.
    cs_caption_x = (cs_xs[0] + cs_xs[-1] + cs_w) / 2
    ax.text(cs_caption_x, ctx_y - 0.22,
            "each drifts only when its storyline is active; frozen otherwise",
            ha="center", va="top", fontsize=7.5,
            style="italic", color="#475569")

    # Highlight the active storyline.
    active_cx = cs_xs[active_k] + cs_w / 2
    ax.text(active_cx, ctx_y + ctx_h + 0.18,
            "$S_2 = S^{*}$ (active)",
            ha="center", va="bottom", fontsize=8,
            style="italic", color="#9f1239")

    # "Frozen" loop-back markers above the inactive stores.
    for k in (0, 2):
        x = cs_xs[k] + cs_w / 2
        ax.add_patch(FancyArrowPatch(
            (x - 0.35, ctx_y + ctx_h + 0.02),
            (x + 0.35, ctx_y + ctx_h + 0.02),
            arrowstyle="-|>", mutation_scale=8,
            linewidth=0.7, color="#64748b",
            connectionstyle="arc3,rad=-0.6",
        ))
        ax.text(x, ctx_y + ctx_h + 0.32, "frozen",
                ha="center", va="bottom", fontsize=7,
                style="italic", color="#64748b")

    # Feature-input -> c_G (with beta_G) and -> c_{S*} (with beta_S).
    fin_x = cin_x
    fin_y = cin_y - 0.25
    # Arrow down to c_G.
    _arrow(ax, fin_x - 0.10, fin_y,
           cg_x + cg_w / 2 + 0.30, ctx_y + ctx_h - 0.02,
           rad=0.25, lw=1.0,
           label="$\\beta_G$", label_offset=(-0.30, 0.10), label_size=9)
    # Arrow down to c_{S*}.
    _arrow(ax, fin_x + 0.10, fin_y,
           active_cx, ctx_y + ctx_h + 0.01,
           rad=-0.10, lw=1.0,
           label="$\\beta_S$", label_offset=(0.30, 0.10), label_size=9)

    # =====================================================================
    # (3) Composite mixing stages
    # =====================================================================
    # Encoding composite on the LEFT, retrieval composite on the RIGHT, so
    # their weight-arrows don't cross each other.
    enc_x, enc_y, enc_w, enc_h = 1.0, 3.40, 5.5, 1.10
    _rbox(ax, enc_x, enc_y, enc_w, enc_h,
          facecolor=COL_COMP, edgecolor="#065f46")
    ax.text(enc_x + enc_w / 2, enc_y + enc_h / 2,
            "encoding composite\n"
            "$\\mathbf{c}_{\\mathrm{comp}}(i) = "
            "w_G\\,\\mathbf{c}_G(t) + w_S\\,\\mathbf{c}_{S^{*}}(t)$",
            ha="center", va="center", fontsize=8.5)

    ret_x, ret_w = 9.0, 6.3
    _rbox(ax, ret_x, enc_y, ret_w, enc_h,
          facecolor=COL_COMP, edgecolor="#065f46")
    ax.text(ret_x + ret_w / 2, enc_y + enc_h / 2,
            "retrieval composite (cue event $j$)\n"
            "$\\mathbf{c}_{\\mathrm{ret}}(j) = "
            "w^{\\mathrm{ret}}_G\\,\\mathbf{c}_G(t_j) + "
            "w^{\\mathrm{ret}}_S\\,\\mathbf{c}_{S_j}(t_j)$",
            ha="center", va="center", fontsize=8.5)
    ax.text(ret_x + ret_w / 2, enc_y + enc_h + 0.18,
            "retrieval weights may differ from encoding weights (task-gated)",
            ha="center", va="bottom", fontsize=7,
            style="italic", color="#475569")

    # c_G -> encoding composite (left).
    _arrow(ax, cg_x + cg_w * 0.35, ctx_y,
           enc_x + enc_w * 0.30, enc_y + enc_h,
           rad=0.05, lw=0.9,
           label="$w_G$", label_offset=(-0.30, 0.0), label_size=9)
    # c_{S*} -> encoding composite.
    _arrow(ax, active_cx - 0.5, ctx_y,
           enc_x + enc_w * 0.75, enc_y + enc_h,
           rad=-0.05, lw=0.9,
           label="$w_S$", label_offset=(0.30, 0.0), label_size=9)

    # c_G -> retrieval composite (right). Route around the encoding box via
    # the top-right of c_G (so the arrow doesn't cross the encoding box).
    _arrow(ax, cg_x + cg_w - 0.05, ctx_y + ctx_h * 0.5,
           ret_x + ret_w * 0.20, enc_y + enc_h,
           rad=-0.30, lw=0.9, color="#475569",
           label="$w_G^{\\mathrm{ret}}$",
           label_offset=(-0.35, 0.10), label_size=8)
    # c_{S_j} -> retrieval composite. (We use c_{S_2} = active one for
    # illustration, matching the active storyline.)
    _arrow(ax, active_cx + 0.8, ctx_y,
           ret_x + ret_w * 0.65, enc_y + enc_h,
           rad=-0.10, lw=0.9, color="#475569",
           label="$w_S^{\\mathrm{ret}}$",
           label_offset=(0.32, 0.10), label_size=8)

    # =====================================================================
    # (4) Decision competition
    # =====================================================================
    dec_y = 0.8
    dec_x, dec_w, dec_h = 1.0, 14.3, 1.70
    _rbox(ax, dec_x, dec_y, dec_w, dec_h,
          facecolor=COL_DECIDE, edgecolor="#a16207")
    ax.text(dec_x + 0.25, dec_y + dec_h - 0.20,
            "decision competition",
            ha="left", va="top", fontsize=9.5, fontweight="bold")
    ax.text(dec_x + 0.25, dec_y + dec_h - 0.55,
            "$s_i = \\mathrm{cos}(\\mathbf{c}_{\\mathrm{ret}}(j),\\,"
            "\\mathbf{c}_{\\mathrm{comp}}(i)) \\cdot a_i "
            "\\cdot e^{-\\lambda I_{ij}}$",
            ha="left", va="top", fontsize=9)
    ax.text(dec_x + 0.25, dec_y + dec_h - 0.95,
            "$P(i \\mid j) = \\mathrm{softmax}_i(s_i)$",
            ha="left", va="top", fontsize=9)

    # Mini pictogram of candidate probabilities on the right side of the
    # decision box.
    cand_x0 = dec_x + dec_w * 0.45
    cand_x1 = dec_x + dec_w - 0.35
    n_cand = 16
    heights = [0.10, 0.15, 0.28, 0.55, 0.32, 0.20, 0.14, 0.10,
               0.08, 0.12, 0.25, 0.40, 0.30, 0.22, 0.15, 0.10]
    step = (cand_x1 - cand_x0) / (n_cand - 1)
    base_y = dec_y + 0.32
    for i, h in enumerate(heights):
        xi = cand_x0 + i * step
        ax.plot([xi, xi], [base_y, base_y + h * 0.9],
                color="#475569", lw=1.0)
        ax.add_patch(mpatches.Circle(
            (xi, base_y + h * 0.9), radius=0.05,
            facecolor="#475569", edgecolor="#1f2937", lw=0.4,
        ))
    winner = heights.index(max(heights))
    xi = cand_x0 + winner * step
    ax.plot([xi, xi], [base_y, base_y + heights[winner] * 0.9],
            color="#dc2626", lw=1.5)
    ax.add_patch(mpatches.Circle(
        (xi, base_y + heights[winner] * 0.9), radius=0.06,
        facecolor="#dc2626", edgecolor="#7f1d1d", lw=0.5,
    ))
    ax.text(xi, base_y + heights[winner] * 0.9 + 0.12,
            "winner",
            ha="center", va="bottom", fontsize=7, color="#dc2626",
            style="italic")
    ax.text(cand_x0 - 0.15, base_y + 0.02,
            "$P(i \\mid j)$ across $i$",
            ha="right", va="center", fontsize=7.5, color="#475569")

    # Encoding composite -> decision (every candidate's c_comp).
    _arrow(ax, enc_x + enc_w / 2, enc_y,
           enc_x + enc_w / 2, dec_y + dec_h,
           lw=0.9,
           label="every candidate's $\\mathbf{c}_{\\mathrm{comp}}(i)$",
           label_offset=(-2.1, 0.0), label_size=7.5)
    # Retrieval composite -> decision.
    _arrow(ax, ret_x + ret_w / 2, enc_y,
           ret_x + ret_w / 2, dec_y + dec_h,
           lw=0.9,
           label="cue $\\mathbf{c}_{\\mathrm{ret}}(j)$",
           label_offset=(1.4, 0.0), label_size=7.5)

    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="paper/figs/source/fig_model.pdf")
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
