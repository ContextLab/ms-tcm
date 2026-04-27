"""Hybrid fitter: trial-LL + weighted curve-matching penalties.

Same fitting process applied to all three models (cz, mstcm, cmr) so
that comparisons are fair. The objective:

    loss(theta) = NLL(theta) + λ · [ w_spc · MSE(SPC) +
                                     w_pfr · MSE(pFR) +
                                     w_crp · MSE(lag_CRP) ]

- NLL is the trial-level negative log-likelihood (closed form for
  cz/cmr, marginalized for mstcm).
- Curves (SPC, pFR, lag-CRP) are computed by simulating n_curve_draws
  synthetic datasets at the current parameters, averaging across draws
  to get aggregate model curves.
- MSE is computed against the observed aggregate curves (computed once).
- λ controls how much curve matching is weighted relative to LL. With
  λ = 50000 (default), curve penalty contributes ~10-25% of total loss
  for typical NLL ~ 10000 and curve RMSE ~ 0.05.
- w_spc, w_pfr, w_crp let the user emphasize specific curves. Default
  is 1.0 each.

Outputs (per --model flag):
  data/processed/fits/{cz_hybrid_frfr, cmr_hybrid_frfr, mstcm_hybrid_frfr}/
      fit_summary.json

Run from repo root:
    PYTHONPATH=code python code/scripts/fit_hybrid.py --model mstcm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MS_TCM_JAX_DTYPE", "float64")

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.optimize import minimize
from scipy.special import expit, logit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from ms_tcm._likelihood_core import (
    simulate_recalls, simulate_recalls_jax_batch as simulate_recalls_jax_batch_cz,
)
from ms_tcm._likelihood_core_cmr import (
    simulate_recalls_cmr, simulate_recalls_cmr_jax_batch,
)
from ms_tcm._likelihood_core_mstcm import (
    compute_list_log_likelihood_mstcm_numpy,
    simulate_recalls_mstcm, simulate_recalls_mstcm_jax_batch,
)
from ms_tcm.dataset import Dataset
from ms_tcm.frfr import load_frfr_category
from ms_tcm.params import ModelParameters

from analyses import lag_crp, pfr, spc as spc_mod


# JAX-grad NLL for cz and cmr (fast).
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ms_tcm._likelihood_core import (
    CoreHyperparams,
    compute_list_log_likelihood,
)
from ms_tcm._likelihood_core_cmr import (
    CMRCoreHyperparams,
    compute_list_log_likelihood_cmr,
)
from ms_tcm._likelihood_core_mstcm import MSCoreHyperparams


# --- Theta parameterizations (model-specific) ---------------------------

# CZ: 7 parameters (same as fit_cz_to_frfr.py)
def theta_to_params_cz(theta):
    eps = 1e-9
    beta_enc = float(np.clip(expit(theta[0]), eps, 1.0 - eps))
    ratio = float(np.clip(expit(theta[1]), eps, 1.0 - eps))
    beta_list = beta_enc * ratio
    beta_list = min(beta_list, beta_enc - 1e-10)
    gamma_fc = float(np.clip(expit(theta[2]), 0.0, 1.0))
    k = float(np.exp(np.clip(theta[3], -30.0, 30.0)))
    beta_rec = float(np.clip(expit(theta[4]), eps, 1.0 - eps))
    epsilon_d = float(np.exp(np.clip(theta[5], -30.0, 30.0)))
    beta_rein = float(np.clip(expit(theta[6]), 0.0, 1.0))
    return ModelParameters(
        beta_enc=beta_enc, beta_story=beta_list, gamma_fc=gamma_fc, k=k,
        beta_rec=beta_rec, epsilon_d=epsilon_d, beta_rein=beta_rein,
        lambda_reinstate=0.0, tau_init=0.0, w_global=0.0,
        paradigm="free_recall",
    )


def params_to_theta_cz(p):
    eps = 1e-6
    theta = np.zeros(7)
    theta[0] = logit(np.clip(p.beta_enc, eps, 1.0 - eps))
    theta[1] = logit(np.clip(p.beta_list / p.beta_enc, eps, 1.0 - eps))
    theta[2] = logit(np.clip(p.gamma_fc, eps, 1.0 - eps))
    theta[3] = np.log(p.k)
    theta[4] = logit(np.clip(p.beta_rec, eps, 1.0 - eps))
    theta[5] = np.log(p.epsilon_d)
    theta[6] = logit(np.clip(p.beta_rein, eps, 1.0 - eps))
    return theta


# CMR: 7 parameters
def theta_to_params_cmr(theta):
    eps = 1e-9
    beta_enc = float(np.clip(expit(theta[0]), eps, 1.0 - eps))
    beta_rec = float(np.clip(expit(theta[1]), eps, 1.0 - eps))
    gamma_fc = float(np.clip(expit(theta[2]), 0.0, 1.0))
    k = float(np.exp(np.clip(theta[3], -30.0, 30.0)))
    epsilon_d = float(np.exp(np.clip(theta[4], -30.0, 30.0)))
    phi_s = float(np.exp(np.clip(theta[5], -30.0, 30.0)))
    phi_d = float(np.exp(np.clip(theta[6], -30.0, 30.0)))
    return ModelParameters(
        beta_enc=beta_enc, beta_story=beta_enc * 0.5, gamma_fc=gamma_fc,
        k=k, beta_rec=beta_rec, epsilon_d=epsilon_d, beta_rein=0.0,
        phi_s=phi_s, phi_d=phi_d,
        lambda_reinstate=0.0, tau_init=0.0, w_global=0.0,
        paradigm="free_recall",
    )


def params_to_theta_cmr(p):
    eps = 1e-6
    theta = np.zeros(7)
    theta[0] = logit(np.clip(p.beta_enc, eps, 1.0 - eps))
    theta[1] = logit(np.clip(p.beta_rec, eps, 1.0 - eps))
    theta[2] = logit(np.clip(p.gamma_fc, eps, 1.0 - eps))
    theta[3] = np.log(p.k)
    theta[4] = np.log(p.epsilon_d)
    theta[5] = np.log(max(p.phi_s, eps))
    theta[6] = np.log(max(p.phi_d, eps))
    return theta


# MS-TCM: 11 parameters
def theta_to_params_mstcm(theta):
    eps = 1e-9
    beta_enc = float(np.clip(expit(theta[0]), eps, 1.0 - eps))
    ratio = float(np.clip(expit(theta[1]), eps, 1.0 - eps))
    beta_list = beta_enc * ratio
    beta_list = min(beta_list, beta_enc - 1e-10)
    gamma_fc = float(np.clip(expit(theta[2]), 0.0, 1.0))
    k = float(np.exp(np.clip(theta[3], -30.0, 30.0)))
    beta_rec = float(np.clip(expit(theta[4]), eps, 1.0 - eps))
    epsilon_d = float(np.exp(np.clip(theta[5], -30.0, 30.0)))
    beta_rein = float(np.clip(expit(theta[6]), 0.0, 1.0))
    beta_enc_g = float(np.clip(expit(theta[7]), eps, 1.0 - eps))
    lam = float(np.clip(expit(theta[8]), 0.0, 1.0))
    tau = float(np.clip(expit(theta[9]), 0.0, 1.0))
    w_global = float(np.clip(expit(theta[10]), 0.0, 1.0))
    return ModelParameters(
        beta_enc=beta_enc, beta_enc_global=beta_enc_g,
        beta_story=beta_list, gamma_fc=gamma_fc, k=k,
        beta_rec=beta_rec, epsilon_d=epsilon_d, beta_rein=beta_rein,
        lambda_reinstate=lam, tau_init=tau, w_global=w_global,
        paradigm="free_recall",
    )


def params_to_theta_mstcm(p):
    eps = 1e-6
    theta = np.zeros(11)
    theta[0] = logit(np.clip(p.beta_enc, eps, 1.0 - eps))
    theta[1] = logit(np.clip(p.beta_list / p.beta_enc, eps, 1.0 - eps))
    theta[2] = logit(np.clip(p.gamma_fc, eps, 1.0 - eps))
    theta[3] = np.log(p.k)
    theta[4] = logit(np.clip(p.beta_rec, eps, 1.0 - eps))
    theta[5] = np.log(p.epsilon_d)
    theta[6] = logit(np.clip(p.beta_rein, eps, 1.0 - eps))
    theta[7] = logit(np.clip(p.beta_enc_global, eps, 1.0 - eps))
    theta[8] = logit(np.clip(p.lambda_reinstate, eps, 1.0 - eps))
    theta[9] = logit(np.clip(p.tau_init, eps, 1.0 - eps))
    theta[10] = logit(np.clip(p.w_global, eps, 1.0 - eps))
    return theta


# --- Per-list inputs ----------------------------------------------------


def build_per_list_inputs_obs(ds: Dataset):
    """Per-list inputs for OBSERVED LL evaluation + curve simulation seed."""
    pdf = ds.presented.to_pandas()
    rdf = ds.recalled.to_pandas()
    keys = sorted(set(zip(pdf["participant"].tolist(), pdf["list"].tolist())))

    ll_inputs = []  # for closed-form LL
    sim_inputs = []  # for simulating recalls
    for part, lst in keys:
        sub_p = pdf[(pdf["participant"]==part) & (pdf["list"]==lst)].sort_values("serial_position")
        sub_r = rdf[(rdf["participant"]==part) & (rdf["list"]==lst)].sort_values("output_position")
        W = len(sub_p)
        cats_seen, cat_to_idx = [], {}
        for c in sub_p["category"].tolist():
            if c not in cat_to_idx:
                cat_to_idx[c] = len(cat_to_idx); cats_seen.append(c)
        K = max(1, len(cats_seen))
        cat_indices = np.array(
            [cat_to_idx[c] for c in sub_p["category"].tolist()], dtype=np.int64,
        )
        sps_obs = sub_r[sub_r["serial_position"]>0]["serial_position"].to_numpy(dtype=np.int64)
        n = len(sps_obs)
        R_pad = max(W, n, 1)
        recall_sps = np.zeros(R_pad, dtype=np.int64)
        recall_mask = np.zeros(R_pad, dtype=bool)
        recall_sps[:n] = sps_obs
        recall_mask[:n] = True
        ll_inputs.append((W, K, cat_indices, recall_sps, recall_mask))
        sim_inputs.append((int(part), int(lst), W, K, cat_indices,
                           sub_p["category"].tolist(),
                           sub_p["word"].tolist(),
                           str(sub_p.iloc[0]["list_group"])))
    return ll_inputs, sim_inputs


# --- JAX-grad NLL builders (cz, cmr) ------------------------------------


def build_jax_nll_cz(ll_inputs):
    Ws = {x[0] for x in ll_inputs}
    assert len(Ws) == 1
    W = next(iter(Ws))
    R_pad = max(x[3].shape[0] for x in ll_inputs)
    n_lists = len(ll_inputs)
    sps_b = np.zeros((n_lists, R_pad), dtype=np.int32)
    mask_b = np.zeros((n_lists, R_pad), dtype=bool)
    for i, (_, _, _, recall_sps, recall_mask) in enumerate(ll_inputs):
        n = recall_sps.shape[0]
        sps_b[i, :n] = recall_sps
        mask_b[i, :n] = recall_mask
    sps_jax = jnp.asarray(sps_b, dtype=jnp.int32)
    mask_jax = jnp.asarray(mask_b, dtype=jnp.bool_)

    from jax.scipy.special import expit as jexpit

    def theta_to_hp_cz(theta):
        beta_enc = jexpit(theta[0])
        ratio = jexpit(theta[1])
        beta_list = beta_enc * ratio
        beta_list = jnp.minimum(beta_list, beta_enc - 1e-10)
        return CoreHyperparams(
            beta_enc=beta_enc, beta_list=beta_list,
            beta_rec=jexpit(theta[4]),
            beta_rein=jexpit(theta[6]),
            gamma_fc=jexpit(theta[2]),
            k=jnp.exp(jnp.clip(theta[3], -30.0, 30.0)),
            epsilon_d=jnp.exp(jnp.clip(theta[5], -30.0, 30.0)),
        )

    def _list_ll(theta, sps, mask):
        return compute_list_log_likelihood(
            theta_to_hp_cz(theta), W=W, recall_sps=sps, recall_mask=mask,
            xp=jnp, dtype=jnp.float64,
        )
    _list_ll_v = jax.vmap(_list_ll, in_axes=(None, 0, 0))

    @jax.jit
    def sum_nll(theta):
        return -jnp.sum(_list_ll_v(theta, sps_jax, mask_jax))

    def nll(theta_np):
        return float(sum_nll(jnp.asarray(theta_np, dtype=jnp.float64)))

    return nll


def build_jax_nll_cmr(ll_inputs):
    Ws = {x[0] for x in ll_inputs}
    assert len(Ws) == 1
    W = next(iter(Ws))
    R_pad = max(x[3].shape[0] for x in ll_inputs)
    n_lists = len(ll_inputs)
    sps_b = np.zeros((n_lists, R_pad), dtype=np.int32)
    mask_b = np.zeros((n_lists, R_pad), dtype=bool)
    for i, (_, _, _, recall_sps, recall_mask) in enumerate(ll_inputs):
        n = recall_sps.shape[0]
        sps_b[i, :n] = recall_sps
        mask_b[i, :n] = recall_mask
    sps_jax = jnp.asarray(sps_b, dtype=jnp.int32)
    mask_jax = jnp.asarray(mask_b, dtype=jnp.bool_)

    from jax.scipy.special import expit as jexpit

    def theta_to_hp_cmr(theta):
        return CMRCoreHyperparams(
            beta_enc=jexpit(theta[0]),
            beta_rec=jexpit(theta[1]),
            gamma_fc=jexpit(theta[2]),
            k=jnp.exp(jnp.clip(theta[3], -30.0, 30.0)),
            epsilon_d=jnp.exp(jnp.clip(theta[4], -30.0, 30.0)),
            phi_s=jnp.exp(jnp.clip(theta[5], -30.0, 30.0)),
            phi_d=jnp.exp(jnp.clip(theta[6], -30.0, 30.0)),
        )

    def _list_ll(theta, sps, mask):
        return compute_list_log_likelihood_cmr(
            theta_to_hp_cmr(theta), W=W, recall_sps=sps, recall_mask=mask,
            xp=jnp, dtype=jnp.float64,
        )
    _list_ll_v = jax.vmap(_list_ll, in_axes=(None, 0, 0))

    @jax.jit
    def sum_nll(theta):
        return -jnp.sum(_list_ll_v(theta, sps_jax, mask_jax))

    def nll(theta_np):
        return float(sum_nll(jnp.asarray(theta_np, dtype=jnp.float64)))

    return nll


def build_numpy_nll_mstcm(ll_inputs):
    def nll(theta_np):
        try:
            params = theta_to_params_mstcm(theta_np)
        except Exception:
            return float("inf")
        total = 0.0
        for (W, K, cat_indices, recall_sps, recall_mask) in ll_inputs:
            ll = compute_list_log_likelihood_mstcm_numpy(
                params, cat_indices, recall_sps, recall_mask, W=W, K=K,
            )
            if not np.isfinite(ll):
                return float("inf")
            total += ll
        return -total
    return nll


# --- Curve simulation + MSE ---------------------------------------------


def _simulate_recall_tensor(model: str, params, sim_inputs, n_draws: int,
                              master_seed: int, W: int):
    """Run JAX-batched sim and return (out, mask, list_to_participant).

    list_to_participant[i] gives the participant id of sim row i (so
    that curves can be aggregated per-participant).
    """
    L = len(sim_inputs)
    N = L * n_draws
    K_set = {x[3] for x in sim_inputs}
    if len(K_set) != 1:
        raise ValueError(f"non-uniform K in sim_inputs: {K_set}")
    K = next(iter(K_set))
    max_recalls = 3 * W

    cat_batch_np = np.zeros((N, W), dtype=np.int32)
    list_to_participant = np.zeros(N, dtype=np.int64)
    for d in range(n_draws):
        for li, (part, _, _, _, cat_indices, *_rest) in enumerate(sim_inputs):
            row = d * L + li
            cat_batch_np[row] = cat_indices
            list_to_participant[row] = part
    cat_batch = jnp.asarray(cat_batch_np)
    keys_batch = jax.random.split(jax.random.PRNGKey(master_seed), N)

    if model == "mstcm":
        hp = MSCoreHyperparams.from_model_parameters(params)
        out, mask = simulate_recalls_mstcm_jax_batch(
            hp, W, K, cat_batch, keys_batch, max_recalls=max_recalls,
        )
    elif model == "cz":
        hp = CoreHyperparams.from_model_parameters(params)
        out, mask = simulate_recalls_jax_batch_cz(
            hp, W, keys_batch, max_recalls=max_recalls,
        )
    elif model == "cmr":
        hp = CMRCoreHyperparams.from_model_parameters(params)
        out, mask = simulate_recalls_cmr_jax_batch(
            hp, W, keys_batch, max_recalls=max_recalls,
        )
    else:
        raise ValueError(f"unknown model: {model!r}")
    return np.asarray(out), np.asarray(mask), list_to_participant


def simulate_curves(model: str, params, sim_inputs, n_draws: int,
                    master_seed: int, ds_template: Dataset, W: int):
    """Aggregate (SPC, pFR, lag-CRP) curves averaged over all sim lists."""
    out, mask, _ = _simulate_recall_tensor(
        model, params, sim_inputs, n_draws, master_seed, W,
    )
    return _curves_from_recall_tensor(out, mask, W)


def simulate_curves_per_participant(
    model: str, params, sim_inputs, n_draws: int,
    master_seed: int, ds_template: Dataset, W: int,
):
    """Per-participant (SPC, pFR, lag-CRP) curves.

    Returns (spc_pp, pfr_pp, crp_pp) of shape (P, W), (P, W), (P, 10).
    Aggregates each participant's `n_draws × n_lists_per_participant`
    simulations into that participant's own curves.
    """
    out, mask, l2p = _simulate_recall_tensor(
        model, params, sim_inputs, n_draws, master_seed, W,
    )
    return per_participant_curves_from_tensor(out, mask, W, l2p)


def _curves_from_recall_tensor(out, mask, W):
    """Wrapper around fit_mstcm_curves.curves_from_recall_tensor."""
    from fit_mstcm_curves import curves_from_recall_tensor
    return curves_from_recall_tensor(out, mask, W)


def per_participant_curves_from_tensor(
    out, mask, W, list_to_participant,
):
    """Per-participant aggregate curves.

    Parameters
    ----------
    out, mask : (N, R) arrays from a JAX-batched simulation.
    W : int — list length.
    list_to_participant : (N,) int — participant index for each sim list.

    Returns
    -------
    spc_pp : (P, W) per-participant SPC.
    pfr_pp : (P, W) per-participant pFR.
    crp_pp : (P, 10) per-participant lag-CRP (|lag|=1..5).
    """
    from fit_mstcm_curves import curves_from_recall_tensor

    list_to_participant = np.asarray(list_to_participant)
    participants = np.unique(list_to_participant)
    P = participants.size
    spc_pp = np.zeros((P, W), dtype=np.float64)
    pfr_pp = np.zeros((P, W), dtype=np.float64)
    crp_pp = np.full((P, 10), np.nan, dtype=np.float64)
    for i, p in enumerate(participants):
        rows = np.where(list_to_participant == p)[0]
        if rows.size == 0:
            continue
        out_p = out[rows]; mask_p = mask[rows]
        s, f, c = curves_from_recall_tensor(out_p, mask_p, W)
        spc_pp[i] = s
        pfr_pp[i] = f
        crp_pp[i] = c
    return spc_pp, pfr_pp, crp_pp


def _curves_from_dataset(ds: Dataset, W: int):
    spc = spc_mod.observed(ds)
    p_fr = pfr.observed(ds)
    crp = lag_crp.observed(ds)
    lags = lag_crp.lag_axis(W)
    keep = (np.abs(lags) >= 1) & (np.abs(lags) <= 5)
    return spc, p_fr, crp[keep]


def per_participant_observed_curves(ds: Dataset, W: int):
    """Per-participant observed (SPC, pFR, lag-CRP) from the recalled table."""
    pdf = ds.presented.to_pandas()
    rdf = ds.recalled.to_pandas()
    parts = sorted(pdf["participant"].unique().tolist())
    P = len(parts)
    spc_pp = np.zeros((P, W), dtype=np.float64)
    pfr_pp = np.zeros((P, W), dtype=np.float64)
    crp_pp = np.full((P, 10), np.nan, dtype=np.float64)

    for i, p in enumerate(parts):
        # Build a Dataset slice for this participant.
        sub_pdf = pdf[pdf["participant"] == p]
        sub_rdf = rdf[rdf["participant"] == p]
        sub_ds = Dataset(
            presented=pa.Table.from_pandas(sub_pdf, preserve_index=False),
            recalled=pa.Table.from_pandas(sub_rdf, preserve_index=False),
            manifest=ds.manifest,
        )
        s, f, c = _curves_from_dataset(sub_ds, W)
        spc_pp[i] = s
        pfr_pp[i] = f
        crp_pp[i] = c
    return spc_pp, pfr_pp, crp_pp


def per_participant_curve_penalty(
    sim_pp_curves, obs_pp_curves, weights,
):
    """Mean-over-participants weighted curve penalty.

    For each participant, compute the same MSE penalty as the aggregate
    version, then average across participants. This treats every
    participant's curves as a target the model should match (a soft
    random-effects style: no per-participant parameters, just per-
    participant evidence).
    """
    spc_sim, pfr_sim, crp_sim = sim_pp_curves
    spc_obs, pfr_obs, crp_obs = obs_pp_curves
    P = spc_sim.shape[0]
    total = 0.0
    for (sim, obs, w) in zip(
        (spc_sim, pfr_sim, crp_sim),
        (spc_obs, pfr_obs, crp_obs),
        weights,
    ):
        diff = sim - obs
        diff = np.where(np.isnan(diff), 0.0, diff)
        # per-participant MSE then mean.
        per_p = (diff ** 2).mean(axis=1)
        total += w * float(per_p.mean())
    return total


def curve_penalty(sim_curves, obs_curves, weights):
    """Sum of weighted MSE per curve (NaN-safe)."""
    total = 0.0
    for (sim, obs, w) in zip(sim_curves, obs_curves, weights):
        diff = sim - obs
        diff = np.where(np.isnan(diff), 0.0, diff)
        total += w * float(np.mean(diff ** 2))
    return total


# --- Main hybrid loss ---------------------------------------------------


def make_hybrid_loss(ds, model: str, lam: float, weights, n_draws: int,
                     curve_seed: int, per_participant: bool = False):
    ll_inputs, sim_inputs = build_per_list_inputs_obs(ds)
    W = ds.num_words_per_list

    if per_participant:
        obs_curves = per_participant_observed_curves(ds, W)
    else:
        obs_curves = _curves_from_dataset(ds, W)

    if model == "cz":
        nll_fn = build_jax_nll_cz(ll_inputs)
        theta_to_params = theta_to_params_cz
    elif model == "cmr":
        nll_fn = build_jax_nll_cmr(ll_inputs)
        theta_to_params = theta_to_params_cmr
    elif model == "mstcm":
        nll_fn = build_numpy_nll_mstcm(ll_inputs)
        theta_to_params = theta_to_params_mstcm
    else:
        raise ValueError(model)

    def _curves(params):
        if per_participant:
            return simulate_curves_per_participant(
                model, params, sim_inputs, n_draws=n_draws,
                master_seed=curve_seed, ds_template=ds, W=W,
            )
        else:
            return simulate_curves(
                model, params, sim_inputs, n_draws=n_draws,
                master_seed=curve_seed, ds_template=ds, W=W,
            )

    def _penalty(sim_curves):
        if per_participant:
            return per_participant_curve_penalty(
                sim_curves, obs_curves, weights,
            )
        else:
            return curve_penalty(sim_curves, obs_curves, weights)

    def loss(theta_np):
        try:
            params = theta_to_params(theta_np)
        except Exception:
            return float("inf")
        try:
            ll_term = nll_fn(theta_np)
        except Exception:
            return float("inf")
        if not np.isfinite(ll_term):
            return float("inf")
        try:
            sim_curves = _curves(params)
        except Exception:
            return float("inf")
        cp = _penalty(sim_curves)
        return ll_term + lam * cp

    def loss_components(theta_np):
        params = theta_to_params(theta_np)
        ll_term = nll_fn(theta_np)
        sim_curves = _curves(params)
        cp = _penalty(sim_curves)
        return ll_term, cp, sim_curves

    return loss, loss_components, theta_to_params


# --- main --------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cz", "mstcm", "cmr"], required=True)
    parser.add_argument("--lambda-curve", type=float, default=50000.0,
                        help="Weight on the curve penalty (vs NLL).")
    parser.add_argument("--w-spc", type=float, default=1.0)
    parser.add_argument("--w-pfr", type=float, default=1.0)
    parser.add_argument("--w-crp", type=float, default=1.0)
    parser.add_argument("--n-draws", type=int, default=3,
                        help="Synthetic dataset replications per loss eval.")
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curve-seed", type=int, default=42,
                        help="Master seed for curve simulations (deterministic).")
    parser.add_argument("--out", default=None)
    parser.add_argument("--restart-std", type=float, default=0.4)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument(
        "--per-participant", action="store_true",
        help="Compute curve penalty per-participant instead of on the "
             "aggregate curve. Better captures heterogeneity in the data.",
    )
    args = parser.parse_args()

    if args.out is None:
        args.out = f"data/processed/fits/{args.model}_hybrid_frfr"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = REPO / f"notes/cz_reproduction/{args.model}_hybrid_fit_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: frfr_category")
    ds = load_frfr_category()
    print(f"  {ds.num_participants} participants × "
          f"{ds.num_lists_per_participant} lists × "
          f"{ds.num_words_per_list} words")

    weights = (args.w_spc, args.w_pfr, args.w_crp)
    print(f"\nBuilding hybrid loss (model={args.model}, λ={args.lambda_curve}, "
          f"weights={weights}, n_draws={args.n_draws})...")
    t0 = time.perf_counter()
    loss, loss_components, theta_to_params = make_hybrid_loss(
        ds, args.model, lam=args.lambda_curve, weights=weights,
        n_draws=args.n_draws, curve_seed=args.curve_seed,
        per_participant=args.per_participant,
    )

    if args.model == "cz":
        p0 = ModelParameters(
            beta_enc=0.679, beta_story=0.40, gamma_fc=0.315, k=6.5,
            beta_rec=0.326, epsilon_d=1.04, beta_rein=0.30,
            lambda_reinstate=0.0, paradigm="free_recall",
        )
        params_to_theta = params_to_theta_cz
        n_theta = 7
    elif args.model == "cmr":
        # Init at Polyn 2009 M62 fit values (φ_s=5.39, φ_d=1.41) so the
        # hybrid optimizer starts from a regime where SPC already has
        # primacy + recency. The curve penalty should keep it nearby.
        p0 = ModelParameters(
            beta_enc=0.745, beta_rec=0.745, gamma_fc=0.581, k=6.5,
            epsilon_d=2.0, phi_s=5.39, phi_d=1.41,
            beta_story=0.4, beta_rein=0.0,
            lambda_reinstate=0.0, tau_init=0.0, w_global=0.0,
            paradigm="free_recall",
        )
        params_to_theta = params_to_theta_cmr
        n_theta = 7
    else:
        p0 = ModelParameters(
            beta_enc=0.7, beta_enc_global=0.4, beta_story=0.30,
            gamma_fc=0.5, k=4.0, beta_rec=0.5, epsilon_d=2.0,
            beta_rein=0.30, lambda_reinstate=0.5, tau_init=0.4,
            w_global=0.2, paradigm="free_recall",
        )
        params_to_theta = params_to_theta_mstcm
        n_theta = 11

    theta_init = params_to_theta(p0)
    print("  warming up...")
    initial_loss = loss(theta_init)
    ll0, cp0, _ = loss_components(theta_init)
    print(f"  warmup ({time.perf_counter()-t0:.1f}s); "
          f"loss={initial_loss:.3f}  NLL={ll0:.3f}  curve_pen={cp0:.6f} "
          f"(λ·cp={args.lambda_curve*cp0:.3f})")

    rng = np.random.default_rng(args.seed)
    log_lines = [
        f"# Hybrid fit ({args.model}) to frfr_category "
        f"(λ={args.lambda_curve}, n_draws={args.n_draws}, "
        f"n_restarts={args.n_restarts}, seed={args.seed})",
        f"loss at init = {initial_loss:.4f} "
        f"(NLL={ll0:.4f}, curve_pen={cp0:.6f})",
    ]

    best_loss = float("inf")
    best_theta = theta_init.copy()
    best_restart = -1
    t_fit = time.perf_counter()
    for r in range(args.n_restarts):
        if r == 0:
            theta0 = theta_init.copy()
        else:
            theta0 = theta_init + rng.normal(0, args.restart_std, size=n_theta)
        t_r = time.perf_counter()
        # Stochastic loss → use larger eps for finite-diff to overcome noise.
        res = minimize(
            loss, theta0, method="L-BFGS-B",
            options={"maxiter": args.maxiter,
                     "ftol": 1e-5, "gtol": 1e-4, "eps": 0.05},
        )
        elapsed = time.perf_counter() - t_r
        ll_star, cp_star, _ = loss_components(res.x)
        log_lines.append(
            f"restart {r}: loss0={loss(theta0):.4f}  loss*={res.fun:.4f}  "
            f"NLL*={ll_star:.4f}  cp*={cp_star:.6f}  iters={res.nit}  "
            f"elapsed={elapsed:.1f}s",
        )
        print(log_lines[-1], flush=True)
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_theta = res.x.copy()
            best_restart = r

    total_t = time.perf_counter() - t_fit
    log_lines.append(
        f"best: loss={best_loss:.4f}  restart={best_restart}  "
        f"total={total_t:.1f}s",
    )
    print(log_lines[-1])
    best_params = theta_to_params(best_theta)
    ll_best, cp_best, sim_curves_best = loss_components(best_theta)
    log_lines.append("")
    log_lines.append(f"Final: NLL={ll_best:.4f}  curve_pen={cp_best:.6f}  "
                     f"λ·cp={args.lambda_curve*cp_best:.4f}")

    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"\nLog: {log_path}")

    summary = {
        "model": f"{args.model}_hybrid_free_recall",
        "model_kind": args.model,
        "dataset": "frfr_category",
        "loss_type": "trial_LL_plus_curve_penalty",
        "lambda_curve": args.lambda_curve,
        "weights": {"spc": args.w_spc, "pfr": args.w_pfr, "crp": args.w_crp},
        "n_draws": args.n_draws,
        "n_restarts": args.n_restarts,
        "seed": args.seed,
        "loss": float(best_loss),
        "log_likelihood": float(-ll_best),
        "negative_log_likelihood": float(ll_best),
        "curve_penalty": float(cp_best),
        "elapsed_seconds": float(total_t),
        "parameters": {
            "beta_enc": float(best_params.beta_enc),
            "beta_enc_global": float(best_params.beta_enc_global),
            "beta_list": float(best_params.beta_list),
            "beta_story": float(best_params.beta_story),
            "gamma_fc": float(best_params.gamma_fc),
            "k": float(best_params.k),
            "beta_rec": float(best_params.beta_rec),
            "epsilon_d": float(best_params.epsilon_d),
            "beta_rein": float(best_params.beta_rein),
            "lambda_reinstate": float(best_params.lambda_reinstate),
            "tau_init": float(best_params.tau_init),
            "w_global": float(best_params.w_global),
            "phi_s": float(best_params.phi_s),
            "phi_d": float(best_params.phi_d),
        },
        "theta_mle": [float(x) for x in best_theta],
    }
    (out_dir / "fit_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"Summary: {out_dir / 'fit_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
