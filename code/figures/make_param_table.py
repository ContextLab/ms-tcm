"""Generate a LaTeX parameter table from fit_summary.json. Writes
paper/figs/source/param_table.tex which the main .tex file \\inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DESCRIPTIONS: dict[str, str] = {
    "beta_global": (
        "Drift rate of the global temporal context. "
        "Higher values mean the global context updates more strongly "
        "toward each new event (and forgets older ones more quickly)."
    ),
    "beta_storyline": (
        "Drift rate of the storyline-specific context. "
        "A value near 0 means each storyline context behaves as a nearly "
        "static signature of the category; a value near 1 means within-"
        "storyline events overwrite one another as they are encoded."
    ),
    "w_global": (
        "Weight on the global context in the composite encoding context. "
        "Higher values make the model more sensitive to global temporal "
        "contiguity (standard TCM sets $w_G = 1$)."
    ),
    "w_storyline": (
        "Weight on the active storyline context in the composite. "
        "Higher values make category or storyline identity a stronger cue "
        "for recall."
    ),
    "w_global_ret": (
        "Global-context weight at retrieval. Can differ from $w_G$ at "
        "encoding, reflecting task-driven reweighting of cues."
    ),
    "w_storyline_ret": (
        "Storyline-context weight at retrieval. Increasing it models an "
        "instruction to recall within the same storyline."
    ),
    "gamma": (
        "Storyline-resumption reinstatement ($\\gamma$, \\S5.1). "
        "0 means no reinstatement; positive values reinstate the prior "
        "storyline context when the same storyline resumes."
    ),
    "lambda_interference": (
        "Differential-interference scaling ($\\lambda$, \\S5.3). "
        "0 disables the mechanism; positive values scale the exponential "
        "interference penalty on retrieval."
    ),
}

SYMBOLS: dict[str, str] = {
    "beta_global": r"$\beta_G$",
    "beta_storyline": r"$\beta_S$",
    "w_global": r"$w_G$",
    "w_storyline": r"$w_S$",
    "w_global_ret": r"$w^{\mathrm{ret}}_G$",
    "w_storyline_ret": r"$w^{\mathrm{ret}}_S$",
    "gamma": r"$\gamma$",
    "lambda_interference": r"$\lambda$",
}

ORDER: tuple[str, ...] = (
    "beta_global", "beta_storyline",
    "w_global", "w_storyline",
    "w_global_ret", "w_storyline_ret",
    "gamma", "lambda_interference",
)


def _fmt_value(info: dict) -> str:
    mle = info["mle"]
    lo = info["ci_lower"]
    hi = info["ci_upper"]
    method = info.get("ci_method", "")
    if method == "derived":
        return f"{mle:.3f} (derived)"
    return f"{mle:.3f} \\,[{lo:.3f},\\,{hi:.3f}]"


def build_table(fit_summary: dict) -> str:
    params = fit_summary["parameters"]
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Maximum-likelihood parameter estimates for MS-TCM "
                 r"fit to the FRFR category condition. "
                 r"Values are MLE $[\text{95\% bootstrap CI}]$ "
                 r"(percentile method, "
                 f"$n_{{\\text{{bootstraps}}}}={fit_summary['parameters']['beta_global']['n_bootstraps']}$)."
                 r" Derived parameters are not fit independently "
                 r"(e.g.\ $w_S = 1 - w_G$).}")
    lines.append(r"\label{tab:params}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}l l l p{0.52\linewidth}@{}}")
    lines.append(r"\hline")
    lines.append(r"Parameter & Symbol & Value $[\text{95\% CI}]$ & Description \\")
    lines.append(r"\hline")
    for name in ORDER:
        if name not in params:
            continue
        info = params[name]
        value = _fmt_value(info)
        sym = SYMBOLS[name]
        desc = DESCRIPTIONS[name]
        pretty_name = name.replace("_", r"\_")
        lines.append(
            rf"\texttt{{{pretty_name}}} & {sym} & {value} & {desc} \\"
        )
    lines.append(r"\hline")
    ll = fit_summary.get("log_likelihood")
    aic = fit_summary.get("aic")
    bic = fit_summary.get("bic")
    n_p = fit_summary.get("n_participants")
    n_l = fit_summary.get("n_lists")
    n_r = fit_summary.get("n_recalls_used")
    lines.append(
        r"\multicolumn{4}{l}{"
        rf"$\log L = {ll:.1f}$, AIC $= {aic:.1f}$, BIC $= {bic:.1f}$; "
        rf"$n_{{\text{{participants}}}}={n_p}$, "
        rf"$n_{{\text{{lists}}}}={n_l}$, "
        rf"$n_{{\text{{recalls}}}}={n_r}$"
        r"} \\"
    )
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", default="data/processed/fits/mstcm")
    parser.add_argument("--out", default="paper/figs/source/param_table.tex")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    fit_file = Path(args.fit, "fit_summary.json")
    if not fit_file.exists():
        print(f"ERROR: {fit_file} not found; run the fit first.")
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force_rerun:
        print(f"{out} exists; use --force-rerun to regenerate")
        return 0

    summary = json.loads(fit_file.read_text())
    tex = build_table(summary)
    out.write_text(tex + "\n")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
