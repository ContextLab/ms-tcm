"""Option A fit-diagnosis via the JAX backend (fast, post-refactor).

The Tier-1 scipy-L-BFGS-B path was too slow to run all 30 participants
across three variants. The post-refactor JAX backend uses analytic
gradients (7x savings per step on 7-dim theta) plus JIT-compiled inner
loops, making the full-dataset sweep tractable.

Variants:
    A1 — Hold epsilon_d fixed at C&Z 2025 default 1.04; fit only the
         shape-relevant parameters (beta_enc, beta_story, gamma_fc, k,
         lambda_reinstate, beta_rec). Hypothesis: eps_d dominates LL
         by trading shape for recall count; fixing it lets the other
         parameters recover shape.

    A2 — Full 7-parameter fit with 10 random restarts drawn from wider
         perturbation (std=1.0 in theta space vs. default 0.3). Tests
         whether the default fitter was stuck in a shape-poor local
         minimum.

    A3 — Dispersed-seed strategy: 5 strategically-chosen parameter
         configurations (all-low, all-high, mid, etc.) each followed
         by L-BFGS-B; take the best. Tests for disjoint basins of
         attraction.

Comparison: log-likelihood + SPC / pFR / lag-CRP shape quality vs.
observed on the full 30-participant FRFR-category dataset.

Run from repo root:
    PYTHONPATH=code MS_TCM_JAX_DTYPE=float64 python notes/fit_diagnosis.py
"""

from __future__ import annotations

import os
os.environ.setdefault("MS_TCM_JAX_DTYPE", "float64")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from ms_tcm.frfr import load_frfr_category
from ms_tcm.dataset import Dataset
from ms_tcm.fit import _params_from_theta, _theta_from_params
from ms_tcm.hcmr import HierarchicalCMRModel, sample_recalls
from ms_tcm.likelihood import dataset_log_likelihood, clear_encoding_cache
from ms_tcm.params import ModelParameters
from scipy.optimize import minimize

from analyses.spc import compute_spc
from analyses.pfr import compute_pfr
from analyses.lag_crp import compute_lag_crp


# --- Shape evaluator (unchanged from Tier-1 version) ---


def evaluate_shape(
    params: ModelParameters, ds: Dataset, n_seeds: int = 20,
    master_seed: int = 42, W: int = 16,
) -> dict:
    """Run n_seeds replications; compute per-curve comparison vs observed."""
    model = HierarchicalCMRModel(params)
    state = model.encode(ds)
    tables = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(master_seed + seed)
        t = sample_recalls(model, ds, rng, state=state)
        df = t.to_pandas()
        df["participant"] = df["participant"] + 100_000 * seed
        tables.append(df)
    combined = pa.Table.from_pandas(pd.concat(tables, ignore_index=True))

    spc_obs = compute_spc(ds, W=W).to_numpy()
    pfr_obs = compute_pfr(ds, W=W).to_numpy()
    crp_obs = compute_lag_crp(ds, W=W)

    spc_sim = compute_spc(combined, W=W).to_numpy()
    pfr_sim = compute_pfr(combined, W=W).to_numpy()
    crp_sim = compute_lag_crp(combined, W=W)

    obs_in = ds.recalled.to_pandas()
    obs_in = obs_in[obs_in["serial_position"] > 0]
    obs_rpl = obs_in.groupby(["participant", "list"]).size().mean()

    sim_in = combined.to_pandas()
    sim_in = sim_in[sim_in["serial_position"] > 0]
    sim_rpl = sim_in.groupby(["participant", "list"]).size().mean()

    key_positions = [0, W // 2, W - 1]
    rel_err = []
    for i in key_positions:
        denom = max(abs(spc_obs[i]), 1e-6)
        rel_err.append(abs(spc_sim[i] - spc_obs[i]) / denom)

    pfr_argmax_sim = int(np.argmax(pfr_sim)) + 1
    pfr_argmax_obs = int(np.argmax(pfr_obs)) + 1

    crp_sim_nz = crp_sim[crp_sim.index != 0]
    crp_obs_nz = crp_obs[crp_obs.index != 0]
    crp_peak_sim = int(crp_sim_nz.idxmax())
    crp_peak_obs = int(crp_obs_nz.idxmax())

    lags_to_show = [-3, -2, -1, 1, 2, 3]
    crp_sim_dict = {lag: float(crp_sim.loc[lag]) for lag in lags_to_show}
    crp_obs_dict = {lag: float(crp_obs.loc[lag]) for lag in lags_to_show}

    return {
        "spc_sim": spc_sim, "spc_obs": spc_obs,
        "pfr_sim": pfr_sim, "pfr_obs": pfr_obs,
        "crp_sim": crp_sim_dict, "crp_obs": crp_obs_dict,
        "sim_recalls_per_list": float(sim_rpl),
        "obs_recalls_per_list": float(obs_rpl),
        "spc_rel_err_key_positions": rel_err,
        "pfr_argmax_sim": pfr_argmax_sim, "pfr_argmax_obs": pfr_argmax_obs,
        "crp_peak_sim": crp_peak_sim, "crp_peak_obs": crp_peak_obs,
    }


# --- JAX-backed fit primitives ---


def _make_jax_nll_grad(ds: Dataset, standard_tcm: bool = False,
                       paradigm: str = "free_recall") -> tuple:
    """Return (nll, grad_nll) for the JAX backend (one compile per dataset)."""
    from ms_tcm.jax_backend.fit_jax import _build_jax_dataset_negloglik
    return _build_jax_dataset_negloglik(
        ds, standard_tcm=standard_tcm, paradigm=paradigm,
        feature_dim=ds.feature_dim, seed=0,
    )


# --- Fit variants ---


def variant_A1(ds: Dataset, seed: int = 42, n_restarts: int = 5) -> dict:
    """Hold epsilon_d fixed at 1.04; fit the other 6 parameters.

    Implementation: we optimize a 6-dim theta and reinsert the fixed
    epsilon_d theta-coordinate in the wrapping objective. The gradient
    from JAX is 7-dim; we just drop the last entry.
    """
    feature_dim = ds.feature_dim
    clear_encoding_cache()
    t0 = time.perf_counter()

    default = ModelParameters(feature_dim=feature_dim)
    theta_init = _theta_from_params(default)
    eps_d_fixed_theta = theta_init[6]  # log(1.04)

    nll7, grad7 = _make_jax_nll_grad(ds)

    def pack(theta6):
        return np.concatenate([theta6, [eps_d_fixed_theta]])

    def obj(theta6):
        return nll7(pack(theta6))

    def jac(theta6):
        g = grad7(pack(theta6))
        return np.asarray(g[:6], dtype=np.float64)

    rng = np.random.default_rng(seed)
    best_nll = np.inf
    best_theta6 = theta_init[:6].copy()
    for r in range(n_restarts):
        if r == 0:
            theta0 = theta_init[:6].copy()
        else:
            theta0 = theta_init[:6] + rng.normal(0, 0.3, size=6)
        res = minimize(obj, theta0, jac=jac, method="L-BFGS-B")
        if res.fun < best_nll:
            best_nll = float(res.fun)
            best_theta6 = res.x.copy()

    best_theta = pack(best_theta6)
    params = _params_from_theta(
        best_theta, standard_tcm=False, paradigm="free_recall",
        feature_dim=feature_dim, seed=seed,
    )
    return {
        "variant": "A1",
        "params": params,
        "log_likelihood": -best_nll,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def variant_A2(ds: Dataset, seed: int = 42, n_restarts: int = 10) -> dict:
    """Full 7-dim fit, wider restart perturbation (std=1.0)."""
    feature_dim = ds.feature_dim
    clear_encoding_cache()
    t0 = time.perf_counter()

    default = ModelParameters(feature_dim=feature_dim)
    theta_init = _theta_from_params(default)
    n_theta = len(theta_init)

    nll, grad = _make_jax_nll_grad(ds)

    rng = np.random.default_rng(seed)
    best_nll = np.inf
    best_theta = theta_init.copy()
    for r in range(n_restarts):
        if r == 0:
            theta0 = theta_init.copy()
        else:
            theta0 = theta_init + rng.normal(0, 1.0, size=n_theta)
        res = minimize(nll, theta0, jac=grad, method="L-BFGS-B")
        if res.fun < best_nll:
            best_nll = float(res.fun)
            best_theta = res.x.copy()

    params = _params_from_theta(
        best_theta, standard_tcm=False, paradigm="free_recall",
        feature_dim=feature_dim, seed=seed,
    )
    return {
        "variant": "A2",
        "params": params,
        "log_likelihood": -best_nll,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def variant_A3(ds: Dataset, seed: int = 42) -> dict:
    """Basin-hopping: 5 dispersed seed points + L-BFGS-B from each."""
    feature_dim = ds.feature_dim
    clear_encoding_cache()
    t0 = time.perf_counter()

    seed_configs = [
        # (beta_enc, beta_story, gamma_fc, k, lambda, beta_rec, eps_d)
        (0.679, 0.400, 0.315, 6.50, 0.80, 0.326, 1.04),   # defaults
        (0.40,  0.20,  0.20,  2.00, 0.20, 0.20,  0.50),   # all-low
        (0.90,  0.45,  0.80,  15.0, 0.95, 0.80,  2.00),   # all-high
        (0.679, 0.400, 0.315, 6.50, 0.50, 0.326, 1.04),   # mid-lambda
        (0.70,  0.30,  0.50,  5.00, 0.70, 0.50,  1.00),   # mid-ish
    ]

    nll, grad = _make_jax_nll_grad(ds)

    best_nll = np.inf
    best_theta = None
    for cfg in seed_configs:
        beta_enc, beta_story, gamma_fc, k, lam, beta_rec, eps_d = cfg
        try:
            p0 = ModelParameters(
                beta_enc=beta_enc,
                beta_story=min(beta_story, beta_enc - 1e-6),
                gamma_fc=gamma_fc, k=k, lambda_reinstate=lam,
                beta_rec=beta_rec, epsilon_d=eps_d, feature_dim=feature_dim,
            )
        except ValueError:
            continue
        theta0 = _theta_from_params(p0)
        res = minimize(nll, theta0, jac=grad, method="L-BFGS-B")
        if res.fun < best_nll:
            best_nll = float(res.fun)
            best_theta = res.x.copy()

    params = _params_from_theta(
        best_theta, standard_tcm=False, paradigm="free_recall",
        feature_dim=feature_dim, seed=seed,
    )
    return {
        "variant": "A3",
        "params": params,
        "log_likelihood": -best_nll,
        "elapsed_seconds": time.perf_counter() - t0,
    }


# --- Main ---


def summarize(
    variant_name: str, params: ModelParameters, ll: float,
    elapsed: float, shape: dict,
) -> None:
    print(f"\n===== {variant_name} =====")
    print(f"elapsed: {elapsed:.1f}s  LL: {ll:.2f}")
    print(f"  beta_enc = {params.beta_enc:.4f}")
    print(f"  beta_story = {params.beta_story:.4f}")
    print(f"  gamma_fc = {params.gamma_fc:.4f}")
    print(f"  k = {params.k:.4f}")
    print(f"  lambda = {params.lambda_reinstate:.4f}")
    print(f"  beta_rec = {params.beta_rec:.4f}")
    print(f"  epsilon_d = {params.epsilon_d:.4f}")
    print(f"  SPC rel err at pos 1/8/16: "
          f"{shape['spc_rel_err_key_positions'][0]:.2f} / "
          f"{shape['spc_rel_err_key_positions'][1]:.2f} / "
          f"{shape['spc_rel_err_key_positions'][2]:.2f}")
    print(f"  pFR argmax: sim={shape['pfr_argmax_sim']} obs={shape['pfr_argmax_obs']}")
    print(f"  CRP peak lag: sim={shape['crp_peak_sim']} obs={shape['crp_peak_obs']}")
    print(f"  recalls/list: sim={shape['sim_recalls_per_list']:.2f} "
          f"obs={shape['obs_recalls_per_list']:.2f}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    ds = load_frfr_category()
    print(f"dataset: {ds.num_participants} participants x "
          f"{ds.num_lists_per_participant} lists x "
          f"{ds.num_words_per_list} words", flush=True)
    print(f"backend: JAX (MS_TCM_JAX_DTYPE={os.environ.get('MS_TCM_JAX_DTYPE', 'float64')})")

    # Baseline.
    print("\n[baseline: C&Z 2025 defaults + v6 lambda=0.80]")
    p_def = ModelParameters()
    ll_def = dataset_log_likelihood(ds, p_def)
    shape_def = evaluate_shape(p_def, ds)
    summarize("DEFAULTS (C&Z 2025 Table 1 + v6 §5 λ=0.80)",
              p_def, ll_def, 0.0, shape_def)

    # A1.
    print("\n[fitting A1: eps_d fixed at 1.04, 5 restarts]")
    res_A1 = variant_A1(ds)
    shape_A1 = evaluate_shape(res_A1["params"], ds)
    summarize("A1 (eps_d fixed, 5 restarts)", res_A1["params"],
              res_A1["log_likelihood"], res_A1["elapsed_seconds"], shape_A1)

    # A2.
    print("\n[fitting A2: 10 restarts, std=1.0 perturbation, full 7-dim]")
    res_A2 = variant_A2(ds, n_restarts=10)
    shape_A2 = evaluate_shape(res_A2["params"], ds)
    summarize("A2 (10 restarts, std=1.0)", res_A2["params"],
              res_A2["log_likelihood"], res_A2["elapsed_seconds"], shape_A2)

    # A3.
    print("\n[fitting A3: 5 dispersed seed points + L-BFGS-B from each]")
    res_A3 = variant_A3(ds)
    shape_A3 = evaluate_shape(res_A3["params"], ds)
    summarize("A3 (basin-hopping, 5 seeds)", res_A3["params"],
              res_A3["log_likelihood"], res_A3["elapsed_seconds"], shape_A3)


if __name__ == "__main__":
    main()
