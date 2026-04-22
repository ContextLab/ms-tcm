"""Figure 2: the Multi-Stream Temporal Context Model architecture.

Laid out in the spirit of Fig. 2 of Polyn, Norman, & Kahana (2009). The
context bank sits at the top, the feature input sits in the middle, the
mixing machinery (analog of CMR's M^FC) sits on the left as an explicit
"encoding composite" box, and the decision-competition block sits at the
bottom. The retrieval composite is rendered inline inside the decision
block (to avoid the arrow congestion that a separate mixing box on the
right introduces): at retrieval the cue is c_ret(j) = w^ret_G c_G(t_j) +
w^ret_S c_{S_j}(t_j), which the decision stage then scores against each
candidate's c_comp(i) via cosine similarity and softmaxes with gain tau.

Substantively MS-TCM differs from CMR:
- the top row is a bank of K + 1 *parallel* context vectors (one global
  c_G + one c_{S_k} per storyline) rather than a single C layer split
  into sub-regions;
- inactive storyline contexts are *frozen*, not merely slow-drifting;
- the retrieval weights (w^ret_G, w^ret_S) may differ from the encoding
  weights (w_G, w_S);
- there are no learned M^FC / M^CF matrices.

Output: paper/figs/source/fig_model.pdf (+ .png preview at 96 dpi).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COL_CONTEXT_G = "#dbeafe"
COL_CONTEXT_S = "#fce7f3"
COL_CONTEXT_A = "#fb7185"
COL_FEATURES  = "#fde68a"
COL_MIX       = "#d1fae5"
COL_DECISION  = "#fef3c7"


def _rbox(ax, x, y, w, h, facecolor="white", edgecolor="#222", lw=0.8, **kw):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.09",
        linewidth=lw, edgecolor=edgecolor, facecolor=facecolor, **kw,
    ))


def _arrow(ax, x1, y1, x2, y2, rad=0.0, color="#222", lw=1.0,
           label=None, label_offset=(0.0, 0.0), label_size=9):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}",
    ))
    if label is not None:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mx + label_offset[0], my + label_offset[1], label,
            ha="center", va="center", fontsize=label_size,
            style="italic",
        )


def _vector_bar(ax, x, y, w, h, label, colour, n_cells=8,
                highlight=False):
    edge = "#be185d" if highlight else "#334155"
    lw = 1.1 if highlight else 0.7
    _rbox(ax, x, y, w, h, facecolor=colour, edgecolor=edge, lw=lw)
    pad = 0.08
    cw = (w - 2 * pad) / n_cells
    ch = h * 0.40
    cy = y + (h - ch) / 2
    for i in range(n_cells):
        ax.add_patch(mpatches.Rectangle(
            (x + pad + i * cw + 0.01, cy), cw - 0.02, ch,
            facecolor="white", edgecolor="#64748b", lw=0.3,
        ))
    ax.text(x + w / 2, y + h + 0.10, label,
            ha="center", va="bottom", fontsize=8.5)


def draw() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    # =================================================================
    # TOP: context bank (K + 1 parallel vectors).
    # =================================================================
    ctx_y = 7.2
    ctx_h = 0.80

    cg_x, cg_w = 2.0, 2.8
    _vector_bar(
        ax, cg_x, ctx_y, cg_w, ctx_h,
        "$\\mathbf{c}_G(t)$  (global)",
        COL_CONTEXT_G,
    )

    cs_w = 2.3
    cs_gap = 0.22
    cs_left = cg_x + cg_w + 0.80
    cs_xs = []
    active_k = 1
    for k in range(3):
        x = cs_left + k * (cs_w + cs_gap)
        cs_xs.append(x)
        _vector_bar(
            ax, x, ctx_y, cs_w, ctx_h,
            f"$\\mathbf{{c}}_{{S_{k+1}}}(t)$",
            COL_CONTEXT_A if k == active_k else COL_CONTEXT_S,
            highlight=(k == active_k),
        )
    active_cx = cs_xs[active_k] + cs_w / 2
    # Active marker to the far right, clear of all arrow paths.
    ax.text(cs_xs[-1] + cs_w + 0.15, ctx_y + ctx_h * 0.5,
            "$S_2 = S^{*}$\n(active)",
            ha="left", va="center", fontsize=7.5, style="italic",
            color="#9f1239")
    # Frozen loop-backs.
    for k in (0, 2):
        x = cs_xs[k] + cs_w / 2
        ax.add_patch(FancyArrowPatch(
            (x - 0.28, ctx_y + ctx_h + 0.04),
            (x + 0.28, ctx_y + ctx_h + 0.04),
            arrowstyle="-|>", mutation_scale=7,
            linewidth=0.7, color="#64748b",
            connectionstyle="arc3,rad=-0.6",
        ))
        ax.text(x, ctx_y + ctx_h + 0.38, "frozen",
                ha="center", va="bottom", fontsize=7,
                style="italic", color="#64748b")

    # =================================================================
    # MIDDLE-LEFT: encoding composite.
    # =================================================================
    enc_x, enc_y = 0.5, 3.6
    enc_w, enc_h = 4.0, 1.80
    _rbox(ax, enc_x, enc_y, enc_w, enc_h,
          facecolor=COL_MIX, edgecolor="#065f46")
    ax.text(enc_x + enc_w / 2, enc_y + enc_h - 0.20,
            "encoding composite",
            ha="center", va="top", fontsize=9, fontweight="bold")
    ax.text(enc_x + enc_w / 2, enc_y + enc_h / 2 - 0.10,
            "$\\mathbf{c}_{\\mathrm{comp}}(i) = "
            "w_G\\,\\mathbf{c}_G(t) + w_S\\,\\mathbf{c}_{S^{*}}(t)$",
            ha="center", va="center", fontsize=9)
    ax.text(enc_x + enc_w / 2, enc_y + 0.20,
            "stored in memory, later\ncompared against the retrieval cue",
            ha="center", va="bottom", fontsize=7,
            style="italic", color="#475569")

    # =================================================================
    # MIDDLE-CENTRE: feature input.
    # =================================================================
    fin_x, fin_y = enc_x + enc_w + 1.2, enc_y + 0.20
    fin_w, fin_h = 4.5, 1.40
    _rbox(ax, fin_x, fin_y, fin_w, fin_h,
          facecolor=COL_FEATURES, edgecolor="#78350f")
    ax.text(fin_x + fin_w / 2, fin_y + fin_h / 2,
            "feature input $\\mathbf{c}^{\\mathrm{IN}}_i$\n"
            "(current event $i$, on storyline $S^{*}$)",
            ha="center", va="center", fontsize=9)
    # "the environment" label above the feature input.
    ax.text(fin_x + fin_w / 2, fin_y + fin_h + 0.30,
            "the environment",
            ha="center", va="bottom", fontsize=9,
            style="italic", color="#475569")
    _arrow(ax, fin_x + fin_w / 2, fin_y + fin_h + 0.28,
           fin_x + fin_w / 2, fin_y + fin_h + 0.02, lw=0.9)

    # =================================================================
    # MIDDLE-RIGHT: retrieval-cue summary.
    # =================================================================
    ret_x, ret_y = fin_x + fin_w + 0.60, enc_y
    ret_w, ret_h = 16 - 0.5 - ret_x, enc_h
    _rbox(ax, ret_x, ret_y, ret_w, ret_h,
          facecolor=COL_MIX, edgecolor="#065f46")
    ax.text(ret_x + ret_w / 2, ret_y + ret_h - 0.20,
            "retrieval cue (cue event $j$)",
            ha="center", va="top", fontsize=9, fontweight="bold")
    ax.text(ret_x + ret_w / 2, ret_y + ret_h / 2 - 0.10,
            "$\\mathbf{c}_{\\mathrm{ret}}(j) = "
            "w^{\\mathrm{ret}}_G\\,\\mathbf{c}_G(t_j)$\n"
            "$\\quad + w^{\\mathrm{ret}}_S\\,\\mathbf{c}_{S_j}(t_j)$",
            ha="center", va="center", fontsize=9)
    ax.text(ret_x + ret_w / 2, ret_y + 0.20,
            "retrieval weights may differ from\nencoding weights (task-gated)",
            ha="center", va="bottom", fontsize=7,
            style="italic", color="#475569")

    # =================================================================
    # Arrows: F -> encoding composite (beta_G, w_G); F -> via c_S (beta_S, w_S)
    # Encoding arrow: feature input enters the contexts (top) with drift
    # rates, and then the contexts feed into the encoding composite.
    # =================================================================
    # feature -> c_G (up-left)
    _arrow(
        ax, fin_x + fin_w * 0.20, fin_y + fin_h,
        cg_x + cg_w * 0.50, ctx_y,
        rad=-0.25, lw=1.0,
        label="$\\beta_G$",
        label_offset=(-0.60, 0.05), label_size=9,
    )
    # feature -> active c_{S*} (up)
    _arrow(
        ax, fin_x + fin_w * 0.80, fin_y + fin_h,
        active_cx, ctx_y,
        rad=0.10, lw=1.0,
        label="$\\beta_S$",
        label_offset=(0.40, 0.20), label_size=9,
    )
    # c_G -> encoding composite (down-left)
    _arrow(
        ax, cg_x + cg_w * 0.30, ctx_y,
        enc_x + enc_w * 0.60, enc_y + enc_h,
        rad=0.25, lw=1.0,
        label="$w_G$",
        label_offset=(-0.40, 0.15), label_size=9,
    )
    # active c_{S*} -> encoding composite (down-left, longer sweep)
    _arrow(
        ax, active_cx, ctx_y,
        enc_x + enc_w * 0.85, enc_y + enc_h,
        rad=0.45, lw=1.0,
        label="$w_S$",
        label_offset=(-0.60, 0.50), label_size=9,
    )
    # c_G -> retrieval cue (down-right)
    _arrow(
        ax, cg_x + cg_w * 0.80, ctx_y,
        ret_x + ret_w * 0.40, ret_y + ret_h,
        rad=-0.35, lw=1.0, color="#475569",
        label="$w^{\\mathrm{ret}}_G$",
        label_offset=(2.60, 0.50), label_size=9,
    )
    # active c_{S*} -> retrieval cue (down-right)
    _arrow(
        ax, active_cx + 0.40, ctx_y,
        ret_x + ret_w * 0.15, ret_y + ret_h,
        rad=-0.25, lw=1.0, color="#475569",
        label="$w^{\\mathrm{ret}}_S$",
        label_offset=(-0.40, 0.30), label_size=9,
    )

    # =================================================================
    # BOTTOM: decision competition.
    # =================================================================
    dec_x, dec_y = 0.5, 0.25
    dec_w, dec_h = 15.0, 2.30
    _rbox(ax, dec_x, dec_y, dec_w, dec_h,
          facecolor=COL_DECISION, edgecolor="#a16207")
    ax.text(dec_x + 0.30, dec_y + dec_h - 0.25,
            "decision competition",
            ha="left", va="top", fontsize=10, fontweight="bold")
    ax.text(dec_x + 0.30, dec_y + dec_h - 0.75,
            "$s_i = \\cos(\\mathbf{c}_{\\mathrm{ret}}(j),\\,"
            "\\mathbf{c}_{\\mathrm{comp}}(i)) \\cdot a_i \\cdot e^{-\\lambda I_{ij}}$",
            ha="left", va="top", fontsize=10)
    ax.text(dec_x + 0.30, dec_y + dec_h - 1.20,
            "$P(i \\mid j) = \\mathrm{softmax}_i(\\tau\\,s_i)$",
            ha="left", va="top", fontsize=10)
    ax.text(dec_x + 0.30, dec_y + dec_h - 1.70,
            "primacy bias: $a_i = \\phi_s\\,e^{-\\phi_d (i-1)} + 1$",
            ha="left", va="top", fontsize=8.5,
            style="italic", color="#475569")

    # Candidate pictogram, right side of the decision box.
    cand_x0 = dec_x + dec_w * 0.55
    cand_x1 = dec_x + dec_w - 0.35
    n_cand = 16
    heights = [0.09, 0.14, 0.28, 0.55, 0.32, 0.20, 0.14, 0.10,
               0.08, 0.12, 0.25, 0.40, 0.30, 0.22, 0.15, 0.10]
    step = (cand_x1 - cand_x0) / (n_cand - 1)
    base_y = dec_y + 0.35
    for i, hi in enumerate(heights):
        xi = cand_x0 + i * step
        ax.plot([xi, xi], [base_y, base_y + hi * 1.4],
                color="#475569", lw=1.0)
        ax.add_patch(mpatches.Circle(
            (xi, base_y + hi * 1.4), radius=0.05,
            facecolor="#475569", edgecolor="#1f2937", lw=0.4,
        ))
    winner = heights.index(max(heights))
    xi_w = cand_x0 + winner * step
    ax.plot([xi_w, xi_w], [base_y, base_y + heights[winner] * 1.4],
            color="#dc2626", lw=1.4)
    ax.add_patch(mpatches.Circle(
        (xi_w, base_y + heights[winner] * 1.4), radius=0.06,
        facecolor="#dc2626", edgecolor="#7f1d1d", lw=0.5,
    ))
    ax.text(xi_w, base_y + heights[winner] * 1.4 + 0.10,
            "winner",
            ha="center", va="bottom", fontsize=7.5,
            style="italic", color="#dc2626")
    ax.text(cand_x0 - 0.15, base_y + 0.05,
            "$P(i \\mid j)$",
            ha="right", va="center", fontsize=8)

    # Downward arrows from the mixing boxes to the decision box.
    _arrow(
        ax, enc_x + enc_w / 2, enc_y,
        enc_x + enc_w / 2, dec_y + dec_h,
        lw=0.9,
        label="candidate $\\mathbf{c}_{\\mathrm{comp}}(i)$",
        label_offset=(1.6, 0.0), label_size=8,
    )
    _arrow(
        ax, ret_x + ret_w / 2, ret_y,
        ret_x + ret_w / 2, dec_y + dec_h,
        lw=0.9,
        label="cue $\\mathbf{c}_{\\mathrm{ret}}(j)$",
        label_offset=(-1.15, 0.0), label_size=8,
    )

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
