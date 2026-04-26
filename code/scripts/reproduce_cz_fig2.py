"""Reproduce Cornell & Zhang 2025 Figure 2b/c by simulating the hierarchical
free-recall model at their published Table 1 parameters and comparing to
observed Kahana et al. 2002 Exp 1 young-adults data.

Outputs:
    - notes/cz_reproduction/sim_vs_obs_kahana2002.png: 3-panel figure
      (SPC, pFR, lag-CRP) with observed (markers) and simulated (lines).
    - Console summary: key shape metrics for both.

Run from repo root:
    PYTHONPATH=code python code/scripts/reproduce_cz_fig2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from ms_tcm.dataset import Dataset, load_dataset
from ms_tcm.params import ModelParameters
from ms_tcm._likelihood_core import simulate_recalls
from analyses.spc import compute_spc
from analyses.pfr import compute_pfr
from analyses.lag_crp import compute_lag_crp


OUT_DIR = REPO / "notes/cz_reproduction"


def make_simulated_dataset(
    ds_observed: Dataset, params: ModelParameters, *, n_seeds: int = 20,
    master_seed: int = 42,
) -> Dataset:
    """Generate a simulated recalled-dataset by running the C&Z sampler.

    For each (participant, list) in the observed Dataset, run ``simulate_recalls``
    and replicate n_seeds times (different participants get renumbered to
    keep IDs unique). The presented side is copied unchanged.
    """
    pdf_obs = ds_observed.presented.to_pandas()
    keys = sorted(set(zip(pdf_obs["participant"].tolist(),
                         pdf_obs["list"].tolist())))
    W = ds_observed.num_words_per_list

    pres_rows = []
    rec_rows = []
    rng_base = np.random.default_rng(master_seed)
    for seed in range(n_seeds):
        for part, lst in keys:
            # Unique pseudo-participant id for each (seed, part).
            pseudo_part = int(part) + 100_000 * seed
            sub_pres = pdf_obs[
                (pdf_obs["participant"] == part) & (pdf_obs["list"] == lst)
            ].sort_values("serial_position").copy()
            sub_pres["participant"] = pseudo_part
            pres_rows.append(sub_pres)

            rng = np.random.default_rng(
                rng_base.integers(0, 2**31 - 1) + seed * 997 + part * 37 + lst,
            )
            recalls = simulate_recalls(params, W=W, rng=rng)
            for out_pos, sp in enumerate(recalls, start=1):
                word_row = sub_pres[sub_pres["serial_position"] == sp].iloc[0]
                rec_rows.append({
                    "participant": pseudo_part,
                    "list": int(lst),
                    "output_position": out_pos,
                    "word": str(word_row["word"]),
                    "category": str(word_row["category"]),
                    "serial_position": int(sp),
                    "list_group": str(word_row["list_group"]),
                })

    pres_df = pd.concat(pres_rows, ignore_index=True)
    rec_df = pd.DataFrame(rec_rows) if rec_rows else pd.DataFrame(
        columns=["participant", "list", "output_position", "word",
                 "category", "serial_position", "list_group"],
    )
    return Dataset(
        presented=pa.Table.from_pandas(pres_df, preserve_index=False),
        recalled=pa.Table.from_pandas(rec_df, preserve_index=False),
        manifest=ds_observed.manifest,
    )


def main() -> None:
    print("Loading Kahana 2002 Exp 1 data...")
    ds_obs = load_dataset(str(REPO / "data/raw/kahana2002"))
    W = ds_obs.num_words_per_list
    print(f"  {ds_obs.num_participants} participants × "
          f"{ds_obs.num_lists_per_participant} lists × {W} words")

    # C&Z 2025 Table 1 published fit values (free recall).
    params = ModelParameters(
        beta_enc=0.679,
        beta_story=0.400,  # = beta_list
        gamma_fc=0.315,
        k=6.50,
        beta_rec=0.326,
        epsilon_d=1.04,
        beta_rein=0.300,
        lambda_reinstate=0.0,  # not used in pure C&Z
        paradigm="free_recall",
    )
    print(f"\nC&Z 2025 Table 1 parameters:")
    for name in ("beta_enc", "beta_list", "beta_rec", "beta_rein",
                 "gamma_fc", "k", "epsilon_d"):
        val = getattr(params, name) if hasattr(params, name) else (
            getattr(params, "beta_story") if name == "beta_list" else None
        )
        print(f"  {name} = {val}")

    print("\nSimulating 20 seeds on all 31×30 lists...")
    ds_sim = make_simulated_dataset(ds_obs, params, n_seeds=20, master_seed=42)
    n_sim_rec = ds_sim.recalled.num_rows
    n_sim_lists = ds_sim.presented.to_pandas()[["participant","list"]].drop_duplicates().shape[0]
    print(f"  simulated {n_sim_rec} recalls across {n_sim_lists} lists")

    print("\nComputing SPC, pFR, lag-CRP for observed + simulated...")
    spc_obs = compute_spc(ds_obs, W=W).to_numpy()
    pfr_obs = compute_pfr(ds_obs, W=W).to_numpy()
    crp_obs = compute_lag_crp(ds_obs, W=W)
    spc_sim = compute_spc(ds_sim, W=W).to_numpy()
    pfr_sim = compute_pfr(ds_sim, W=W).to_numpy()
    crp_sim = compute_lag_crp(ds_sim, W=W)

    # Summary metrics.
    rec_obs = ds_obs.recalled.to_pandas()
    rec_sim = ds_sim.recalled.to_pandas()
    obs_rpl = rec_obs[rec_obs["serial_position"] > 0].groupby(
        ["participant", "list"]).size().mean()
    sim_rpl = rec_sim[rec_sim["serial_position"] > 0].groupby(
        ["participant", "list"]).size().mean()

    print("\n=== Shape metrics ===")
    print(f"{'metric':<30} {'observed':>12} {'simulated':>12}")
    print(f"{'SPC @ pos 1 (primacy)':<30} {spc_obs[0]:>12.3f} {spc_sim[0]:>12.3f}")
    print(f"{'SPC @ pos W/2':<30} {spc_obs[W//2]:>12.3f} {spc_sim[W//2]:>12.3f}")
    print(f"{'SPC @ pos W (recency)':<30} {spc_obs[-1]:>12.3f} {spc_sim[-1]:>12.3f}")
    print(f"{'SPC mean':<30} {spc_obs.mean():>12.3f} {spc_sim.mean():>12.3f}")
    print(f"{'pFR argmax':<30} {int(np.argmax(pfr_obs))+1:>12d} {int(np.argmax(pfr_sim))+1:>12d}")
    print(f"{'pFR @ last (recency)':<30} {pfr_obs[-1]:>12.3f} {pfr_sim[-1]:>12.3f}")
    print(f"{'lag-CRP peak lag':<30} {int(crp_obs[crp_obs.index != 0].idxmax()):>12d} "
          f"{int(crp_sim[crp_sim.index != 0].idxmax()):>12d}")
    print(f"{'lag-CRP +1':<30} {crp_obs.loc[1]:>12.3f} {crp_sim.loc[1]:>12.3f}")
    print(f"{'lag-CRP -1':<30} {crp_obs.loc[-1]:>12.3f} {crp_sim.loc[-1]:>12.3f}")
    print(f"{'recalls/list':<30} {obs_rpl:>12.2f} {sim_rpl:>12.2f}")

    # --- Figure ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=100)
    positions = np.arange(1, W + 1)

    ax = axes[0]
    ax.plot(positions, spc_obs, "ko-", label="observed (Kahana 2002)", markersize=6)
    ax.plot(positions, spc_sim, "r-", label="C&Z sim (Table 1 params)", linewidth=2)
    ax.set_xlabel("Serial position")
    ax.set_ylabel("P(recall)")
    ax.set_title("Serial Position Curve (SPC)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(positions, pfr_obs, "ko-", label="observed", markersize=6)
    ax.plot(positions, pfr_sim, "r-", label="C&Z sim", linewidth=2)
    ax.set_xlabel("Serial position")
    ax.set_ylabel("P(first recall)")
    ax.set_title("Probability of First Recall (pFR)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    lags = [l for l in crp_obs.index if l != 0 and -5 <= l <= 5]
    crp_obs_vals = [crp_obs.loc[l] for l in lags]
    crp_sim_vals = [crp_sim.loc[l] if l in crp_sim.index else np.nan for l in lags]
    ax.plot(lags, crp_obs_vals, "ko-", label="observed", markersize=6)
    ax.plot(lags, crp_sim_vals, "r-", label="C&Z sim", linewidth=2)
    ax.set_xlabel("Lag")
    ax.set_ylabel("P(transition)")
    ax.set_title("Lag-CRP")
    ax.axvline(0, color="gray", alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"C&Z 2025 hierarchical CMR at Table 1 parameters vs Kahana 2002 Exp 1 (W={W})",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = OUT_DIR / "sim_vs_obs_kahana2002.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")


if __name__ == "__main__":
    main()
