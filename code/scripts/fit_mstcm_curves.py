"""Fit MS-TCM or C&Z to FRFR-category by minimizing curve RMSE.

Alternative fitting objective to ``fit_*_to_frfr.py`` (trial-LL).
Supports BOTH models via the ``--model`` flag so the comparison is
fair (both fit under the same objective).

Procedure:
  1. Simulates the FRFR-category dataset N times at each parameter
     setting (using a fixed master seed so the loss is deterministic
     in the parameters).
  2. Computes mean SPC, pFR, lag-CRP across simulations.
  3. Computes RMSE vs observed curves, treating each curve as a flat
     feature vector and summing squared errors equally.
  4. Returns total RMSE; scipy L-BFGS-B minimizes via finite differences
     (with `eps=0.05` to overcome stochastic-loss noise floor).

This matches C&Z 2025's methodology (Bayesian-optimization on RMSE
between simulated and observed curves; see their Methods §"Model
Fitting and Simulations").

Output:
    --model=mstcm: data/processed/fits/mstcm_curves_frfr/fit_summary.json
    --model=cz:    data/processed/fits/cz_curves_frfr/fit_summary.json

Run from repo root:
    PYTHONPATH=code python code/scripts/fit_mstcm_curves.py \\
        --model mstcm --n-draws 5 --n-restarts 3 --seed 42
    PYTHONPATH=code python code/scripts/fit_mstcm_curves.py \\
        --model cz    --n-draws 5 --n-restarts 3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.optimize import minimize
from scipy.special import expit, logit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from ms_tcm._likelihood_core import simulate_recalls
from ms_tcm._likelihood_core_cmr import simulate_recalls_cmr
from ms_tcm._likelihood_core_mstcm import (
    MSCoreHyperparams,
    simulate_recalls_mstcm,
    simulate_recalls_mstcm_jax_batch,
)
from ms_tcm.dataset import Dataset
from ms_tcm.frfr import load_frfr_category
from ms_tcm.params import ModelParameters

from analyses import lag_crp, pfr, spc as serial_position

# Lazy JAX import — only when --jax-batch is on, so default path keeps
# the existing numpy-only behavior available.
try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
    jax.config.update("jax_enable_x64", True)
except ImportError:
    jax = None  # type: ignore
    jnp = None  # type: ignore
    _JAX_AVAILABLE = False


# --- Model-specific theta parameterizations ---
#
# C&Z baseline: 7 free parameters
#   theta[0..6] = (beta_enc, beta_list_ratio, gamma_fc, log_k, beta_rec,
#                  log_epsilon_d, beta_rein)
# MS-TCM: 11 free parameters
#   C&Z + (beta_enc_global, lambda_reinstate, tau_init, w_global) at
#   theta[7..10].

N_THETA_CZ = 7
N_THETA_MSTCM = 11
N_THETA_CMR = 7  # beta_enc, beta_rec, gamma_fc, log_k, log_epsilon_d, phi_s, log(phi_d)


def theta_to_params_cmr(theta) -> ModelParameters:
    """Map 7-d theta -> ModelParameters for Polyn 2009 standard CMR.

    Parameterization choices:
      - phi_s ∈ [0, ∞): unconstrained (>=0); use exp(theta[5]) to keep
        positive while allowing 0 limit (= no primacy boost).
      - phi_d ∈ (0, ∞): exp(theta[6]).
      - beta_list, beta_rein not used; pass dummy values.
    """
    eps = 1e-9
    beta_enc = float(np.clip(expit(theta[0]), eps, 1.0 - eps))
    beta_rec = float(np.clip(expit(theta[1]), eps, 1.0 - eps))
    gamma_fc = float(np.clip(expit(theta[2]), 0.0, 1.0))
    k = float(np.exp(np.clip(theta[3], -30.0, 30.0)))
    epsilon_d = float(np.exp(np.clip(theta[4], -30.0, 30.0)))
    phi_s = float(np.exp(np.clip(theta[5], -30.0, 30.0)))
    phi_d = float(np.exp(np.clip(theta[6], -30.0, 30.0)))
    return ModelParameters(
        beta_enc=beta_enc, beta_story=beta_enc * 0.5,  # unused but required
        gamma_fc=gamma_fc, k=k, beta_rec=beta_rec, epsilon_d=epsilon_d,
        beta_rein=0.0, lambda_reinstate=0.0, tau_init=0.0, w_global=0.0,
        phi_s=phi_s, phi_d=phi_d, paradigm="free_recall",
    )


def params_to_theta_cmr(p: ModelParameters) -> np.ndarray:
    eps = 1e-6
    theta = np.zeros(N_THETA_CMR, dtype=np.float64)
    theta[0] = logit(np.clip(p.beta_enc, eps, 1.0 - eps))
    theta[1] = logit(np.clip(p.beta_rec, eps, 1.0 - eps))
    theta[2] = logit(np.clip(p.gamma_fc, eps, 1.0 - eps))
    theta[3] = np.log(p.k)
    theta[4] = np.log(p.epsilon_d)
    theta[5] = np.log(max(p.phi_s, eps))
    theta[6] = np.log(max(p.phi_d, eps))
    return theta


def theta_to_params_cz(theta) -> ModelParameters:
    """Map 7-d theta -> ModelParameters for C&Z (no lambda, no tau, no w_global)."""
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


def params_to_theta_cz(p: ModelParameters) -> np.ndarray:
    eps = 1e-6
    theta = np.zeros(N_THETA_CZ, dtype=np.float64)
    theta[0] = logit(np.clip(p.beta_enc, eps, 1.0 - eps))
    theta[1] = logit(np.clip(p.beta_list / p.beta_enc, eps, 1.0 - eps))
    theta[2] = logit(np.clip(p.gamma_fc, eps, 1.0 - eps))
    theta[3] = np.log(p.k)
    theta[4] = logit(np.clip(p.beta_rec, eps, 1.0 - eps))
    theta[5] = np.log(p.epsilon_d)
    theta[6] = logit(np.clip(p.beta_rein, eps, 1.0 - eps))
    return theta


def theta_to_params_mstcm(theta) -> ModelParameters:
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


def params_to_theta_mstcm(p: ModelParameters) -> np.ndarray:
    eps = 1e-6
    theta = np.zeros(N_THETA_MSTCM, dtype=np.float64)
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


def build_per_list_inputs(ds: Dataset):
    """Per-list inputs needed by the simulator."""
    pdf = ds.presented.to_pandas()
    keys = sorted(set(zip(pdf["participant"].tolist(), pdf["list"].tolist())))
    out = []
    for part, lst in keys:
        sub_p = pdf[(pdf["participant"] == part) & (pdf["list"] == lst)].sort_values("serial_position")
        W = len(sub_p)
        cats_seen: list[str] = []
        cat_to_idx: dict[str, int] = {}
        for c in sub_p["category"].tolist():
            if c not in cat_to_idx:
                cat_to_idx[c] = len(cat_to_idx)
                cats_seen.append(c)
        K = max(1, len(cats_seen))
        cat_indices = np.array(
            [cat_to_idx[c] for c in sub_p["category"].tolist()], dtype=np.int64,
        )
        out.append((int(part), int(lst), W, K, cat_indices,
                    sub_p["category"].tolist(),
                    sub_p["word"].tolist(),
                    str(sub_p.iloc[0]["list_group"])))
    return out


def curves_from_recall_tensor(
    out: np.ndarray,  # (N, R) int32; sps 1-indexed; 0 = padding
    mask: np.ndarray,  # (N, R) bool
    W: int,
):
    """Compute (SPC, pFR, lag-CRP±1..±5) directly from a recall tensor.

    Mirrors the canonical analyses (`analyses.spc.observed`,
    `analyses.pfr.observed`, `analyses.lag_crp.observed`) without
    going through a Dataset/DataFrame, since materializing those for
    every loss evaluation is the bottleneck of the JAX-batched fitter.

    Returns a single concatenated 1-D feature vector matching
    ``compute_curves(...)``: [SPC(W) | pFR(W) | lag-CRP(±1..±5)].
    """
    N, R = out.shape
    valid = mask & (out > 0)

    # SPC: probability that each serial position is recalled at least once.
    # For each list, mark which sps appear; mean across lists.
    # Vectorized: build one-hot of out then mask out invalid then OR-reduce.
    # Use np.add.at for sparse accumulation; faster than scatter for small R.
    spc = np.zeros(W, dtype=np.float64)
    # Per-row, count distinct sps recalled.
    # Use the "max over R" trick: list i recalled sp v if any (out[i, :], v).
    # We build a (N, W) indicator using bincount per row — slow in Python loop;
    # vectorize with a flat scatter and per-row uniqueness via sort+diff.
    # Simpler: for each sp v in 1..W, did any out[i, :]==v with valid?
    for v in range(1, W + 1):
        present = (out == v) & valid  # (N, R)
        spc[v - 1] = np.any(present, axis=1).mean()

    # pFR: distribution of FIRST recall over sps.
    # For each list, find the first valid recall.
    # If no valid recalls, omit (no contribution).
    first_idx = np.argmax(valid, axis=1)  # (N,) — first True; 0 if all False
    has_any = np.any(valid, axis=1)  # (N,)
    first_sps = out[np.arange(N), first_idx]  # (N,) sp of first recall (or 0)
    pfr = np.zeros(W, dtype=np.float64)
    # Histogram over lists with at least one recall.
    valid_first_sps = first_sps[has_any]
    if valid_first_sps.size > 0:
        np.add.at(pfr, valid_first_sps - 1, 1.0)
        pfr /= N  # normalize by total lists (matches `pfr.observed`)

    # lag-CRP±1..±5: averaged over consecutive recall pairs across all lists.
    # For each adjacent pair (out[i, j], out[i, j+1]) where both are valid,
    # contribute 1 to numerator at lag = sp2 - sp1, and 1 to each
    # "possible lag" denominator (lags toward sps not yet recalled at that
    # transition). Reuse the same definition as analyses.lag_crp.
    lag_max = 5
    crp_num = np.zeros(2 * lag_max + 1, dtype=np.float64)
    crp_den = np.zeros(2 * lag_max + 1, dtype=np.float64)
    # iterate per row.
    for i in range(N):
        valid_i = valid[i]
        sps = out[i, valid_i]  # 1-indexed sps in recall order
        if sps.size < 2:
            continue
        recalled_so_far = np.zeros(W + 1, dtype=bool)  # 1-indexed
        recalled_so_far[sps[0]] = True
        for j in range(1, sps.size):
            sp_prev = int(sps[j - 1])
            sp_cur = int(sps[j])
            if recalled_so_far[sp_cur]:
                # Repeat — skip both numerator and denominator (matches
                # canonical analysis which counts only first recall of each).
                continue
            actual_lag = sp_cur - sp_prev
            if -lag_max <= actual_lag <= lag_max and actual_lag != 0:
                crp_num[actual_lag + lag_max] += 1.0
            # Denominator: every possible non-recalled sp at this transition.
            for v in range(1, W + 1):
                if recalled_so_far[v]:
                    continue
                lag = v - sp_prev
                if -lag_max <= lag <= lag_max and lag != 0:
                    crp_den[lag + lag_max] += 1.0
            recalled_so_far[sp_cur] = True
    with np.errstate(divide="ignore", invalid="ignore"):
        crp_full = np.where(crp_den > 0, crp_num / crp_den, np.nan)
    # Drop the lag=0 bin to match `compute_curves` (keeps |lag| 1..5).
    keep = np.abs(np.arange(-lag_max, lag_max + 1)) >= 1
    crp = crp_full[keep]
    return spc, pfr, crp


def simulate_recalls_tensor_mstcm_jax(
    params: ModelParameters, *,
    n_draws: int, master_seed: int, list_inputs,
):
    """Run ONE batched JAX simulator call; return (out, mask, W).

    Skips dataset/dataframe assembly entirely — appropriate when the
    only downstream use is curve computation. Used by the
    curve-matching loss to avoid the pandas overhead that dominates
    each loss evaluation.
    """
    if not _JAX_AVAILABLE:
        raise RuntimeError("JAX backend not available")
    Ws = {W for _, _, W, _, _, _, _, _ in list_inputs}
    Ks = {K for _, _, _, K, _, _, _, _ in list_inputs}
    if len(Ws) != 1 or len(Ks) != 1:
        raise ValueError(
            f"JAX batched path requires uniform W,K across lists; got W={Ws}, K={Ks}",
        )
    W = next(iter(Ws))
    K = next(iter(Ks))
    L = len(list_inputs)
    N = L * n_draws
    max_recalls = 3 * W
    cat_batch = np.zeros((N, W), dtype=np.int32)
    for d_idx in range(n_draws):
        for li, (part, lst, _, _, cat_indices, *_rest) in enumerate(list_inputs):
            cat_batch[d_idx * L + li] = cat_indices
    cat_batch_j = jnp.asarray(cat_batch)
    keys_batch = jax.random.split(jax.random.PRNGKey(master_seed), N)
    hp = MSCoreHyperparams.from_model_parameters(params)
    out_batch, mask_batch = simulate_recalls_mstcm_jax_batch(
        hp, W, K, cat_batch_j, keys_batch, max_recalls=max_recalls,
    )
    return np.asarray(out_batch), np.asarray(mask_batch), W


def simulate_dataset_mstcm_jax_batched(
    ds: Dataset, params: ModelParameters, *,
    n_draws: int, master_seed: int,
    list_inputs,
) -> Dataset:
    """Simulate the full dataset using ONE batched JAX simulator call.

    Shape requirements: all lists must have the same W and K. (FRFR-category
    is uniformly W=16, K=4.) Falls back with a clear error otherwise.
    """
    if not _JAX_AVAILABLE:
        raise RuntimeError("JAX backend not available")

    pdf_orig = ds.presented.to_pandas()
    Ws = {W for _, _, W, _, _, _, _, _ in list_inputs}
    Ks = {K for _, _, _, K, _, _, _, _ in list_inputs}
    if len(Ws) != 1 or len(Ks) != 1:
        raise ValueError(
            f"JAX batched path requires uniform W,K across lists; got W={Ws}, K={Ks}",
        )
    W = next(iter(Ws))
    K = next(iter(Ks))
    L = len(list_inputs)
    N = L * n_draws  # total per-list-per-draw simulations
    max_recalls = 3 * W

    # Build batched cat_indices: shape (N, W). Order: outer=draw, inner=list.
    cat_batch = np.zeros((N, W), dtype=np.int32)
    list_index_lookup = []
    for d_idx in range(n_draws):
        for li, (part, lst, _, _, cat_indices, cats, words, list_group) in enumerate(list_inputs):
            row = d_idx * L + li
            cat_batch[row] = cat_indices
            list_index_lookup.append((d_idx, li, part, lst, cats, words, list_group))

    cat_batch_j = jnp.asarray(cat_batch)
    keys_batch = jax.random.split(jax.random.PRNGKey(master_seed), N)

    hp = MSCoreHyperparams.from_model_parameters(params)
    out_batch, mask_batch = simulate_recalls_mstcm_jax_batch(
        hp, W, K, cat_batch_j, keys_batch, max_recalls=max_recalls,
    )
    out = np.asarray(out_batch)  # (N, max_recalls) int32, sps 1-indexed; 0=pad
    mask = np.asarray(mask_batch)

    pres_rows = []
    rec_rows = []
    for row, (d_idx, li, part, lst, cats, words, list_group) in enumerate(list_index_lookup):
        pseudo_part = part + 100_000 * d_idx
        sub_pres = pdf_orig[
            (pdf_orig["participant"] == part) & (pdf_orig["list"] == lst)
        ].sort_values("serial_position").copy()
        sub_pres["participant"] = pseudo_part
        pres_rows.append(sub_pres)

        valid_sps = out[row, mask[row]]
        for out_pos, sp in enumerate(valid_sps, start=1):
            sp_i = int(sp)
            if sp_i == 0:
                continue
            rec_rows.append({
                "participant": pseudo_part,
                "list": lst,
                "output_position": out_pos,
                "word": words[sp_i - 1],
                "category": cats[sp_i - 1],
                "serial_position": sp_i,
                "list_group": list_group,
            })

    pres_df = pd.concat(pres_rows, ignore_index=True)
    rec_df = pd.DataFrame(rec_rows) if rec_rows else pd.DataFrame(
        columns=["participant", "list", "output_position", "word",
                 "category", "serial_position", "list_group"],
    )
    return Dataset(
        presented=pa.Table.from_pandas(pres_df, preserve_index=False),
        recalled=pa.Table.from_pandas(rec_df, preserve_index=False),
        manifest=ds.manifest,
    )


def simulate_dataset(
    ds: Dataset, params: ModelParameters, *,
    n_draws: int, master_seed: int,
    list_inputs,
    model: str,
) -> Dataset:
    """Simulate the entire FRFR-category dataset n_draws times.

    ``model`` selects the simulator:
      "cz"    — single-storyline C&Z hierarchical (uses simulate_recalls)
      "mstcm" — multi-storyline MS-TCM (uses simulate_recalls_mstcm with
                cat_indices)
    """
    pdf_orig = ds.presented.to_pandas()
    pres_rows = []
    rec_rows = []

    rng_master = np.random.default_rng(master_seed)
    for draw in range(n_draws):
        for part, lst, W, K, cat_indices, cats, words, list_group in list_inputs:
            pseudo_part = part + 100_000 * draw
            sub_pres = pdf_orig[
                (pdf_orig["participant"] == part) & (pdf_orig["list"] == lst)
            ].sort_values("serial_position").copy()
            sub_pres["participant"] = pseudo_part
            pres_rows.append(sub_pres)

            seed = int(rng_master.integers(0, 2**31 - 1))
            seed = seed ^ (draw * 997) ^ (part * 37) ^ lst
            rng = np.random.default_rng(seed)
            if model == "mstcm":
                recalls = simulate_recalls_mstcm(
                    params, W=W, K=K, cat_indices=cat_indices, rng=rng,
                )
            elif model == "cz":
                recalls = simulate_recalls(params, W=W, rng=rng)
            elif model == "cmr":
                recalls = simulate_recalls_cmr(params, W=W, rng=rng)
            else:
                raise ValueError(f"unknown model: {model!r}")

            for out_pos, sp in enumerate(recalls, start=1):
                rec_rows.append({
                    "participant": pseudo_part,
                    "list": lst,
                    "output_position": out_pos,
                    "word": words[sp - 1],
                    "category": cats[sp - 1],
                    "serial_position": int(sp),
                    "list_group": list_group,
                })

    pres_df = pd.concat(pres_rows, ignore_index=True)
    rec_df = pd.DataFrame(rec_rows) if rec_rows else pd.DataFrame(
        columns=["participant", "list", "output_position", "word",
                 "category", "serial_position", "list_group"],
    )
    return Dataset(
        presented=pa.Table.from_pandas(pres_df, preserve_index=False),
        recalled=pa.Table.from_pandas(rec_df, preserve_index=False),
        manifest=ds.manifest,
    )


def compute_curves(ds: Dataset, W: int):
    """Return (spc, pfr, crp) numpy arrays. NaN-cleaned."""
    spc = serial_position.observed(ds)
    p_fr = pfr.observed(ds)
    crp = lag_crp.observed(ds)
    # CRP is indexed by lag (-W+1 to W-1); clip to ±5 lags.
    lags = lag_crp.lag_axis(W)
    keep = (np.abs(lags) >= 1) & (np.abs(lags) <= 5)
    crp_useful = crp[keep]
    return spc, p_fr, crp_useful


def make_loss(
    ds: Dataset, n_draws: int, master_seed: int, model: str,
    *, use_jax_batch: bool = False,
):
    """Build the curve-matching loss function for the given model."""
    W = ds.num_words_per_list
    list_inputs = build_per_list_inputs(ds)
    theta_to_params = {
        "mstcm": theta_to_params_mstcm,
        "cz": theta_to_params_cz,
        "cmr": theta_to_params_cmr,
    }[model]

    obs_spc, obs_pfr, obs_crp = compute_curves(ds, W)
    obs_vec = np.concatenate([obs_spc, obs_pfr, obs_crp])

    print(f"  Observed feature vector length: {len(obs_vec)} "
          f"(SPC={len(obs_spc)} + pFR={len(obs_pfr)} + lag-CRP={len(obs_crp)})")

    if use_jax_batch and model != "mstcm":
        # The JAX batched simulator currently only implements MS-TCM.
        # For C&Z, fall back to numpy.
        use_jax_batch = False

    if use_jax_batch:
        print(f"  Using JAX batched simulator (single vmap'd call per loss eval).")

    def loss(theta_np: np.ndarray) -> float:
        try:
            params = theta_to_params(theta_np)
        except Exception:
            return float("inf")
        try:
            if use_jax_batch:
                # Skip dataframe assembly; compute curves directly.
                out, mask, W_check = simulate_recalls_tensor_mstcm_jax(
                    params, n_draws=n_draws, master_seed=master_seed,
                    list_inputs=list_inputs,
                )
                sim_spc, sim_pfr, sim_crp = curves_from_recall_tensor(
                    out, mask, W_check,
                )
            else:
                sim_ds = simulate_dataset(
                    ds, params, n_draws=n_draws, master_seed=master_seed,
                    list_inputs=list_inputs, model=model,
                )
                sim_spc, sim_pfr, sim_crp = compute_curves(sim_ds, W)
        except Exception:
            return float("inf")
        sim_vec = np.concatenate([sim_spc, sim_pfr, sim_crp])
        diff = sim_vec - obs_vec
        diff = np.where(np.isnan(diff), 0.0, diff)
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        return rmse

    return loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cz", "mstcm", "cmr"], default="mstcm",
                        help="Which model to fit (default: mstcm). "
                             "cmr = Polyn et al. 2009 standard CMR baseline.")
    parser.add_argument("--n-draws", type=int, default=5,
                        help="Synthetic-dataset replications per loss eval.")
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None,
                        help="Output dir; auto-derived from --model if omitted.")
    parser.add_argument("--restart-std", type=float, default=0.4)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument(
        "--jax-batch", action="store_true",
        help="Use JAX batched simulator (mstcm only; ~10-50× speedup).",
    )
    args = parser.parse_args()

    if args.out is None:
        args.out = {
            "cz": "data/processed/fits/cz_curves_frfr",
            "mstcm": "data/processed/fits/mstcm_curves_frfr",
            "cmr": "data/processed/fits/cmr_curves_frfr",
        }[args.model]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = REPO / f"notes/cz_reproduction/{args.model}_curves_fit_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset: frfr_category")
    ds = load_frfr_category()
    print(f"  {ds.num_participants} participants × {ds.num_lists_per_participant} lists × "
          f"{ds.num_words_per_list} words")

    print(f"\nBuilding curve-matching loss for model={args.model} "
          f"(n_draws={args.n_draws}, master_seed={args.seed})...")
    t0 = time.perf_counter()
    loss = make_loss(
        ds, n_draws=args.n_draws, master_seed=args.seed, model=args.model,
        use_jax_batch=args.jax_batch,
    )

    # Initial parameters per model.
    if args.model == "cmr":
        # Polyn 2009 Table 1 fit values (approx).
        p0 = ModelParameters(
            beta_enc=0.74, beta_story=0.40, gamma_fc=0.43, k=4.0,
            beta_rec=0.61, epsilon_d=1.0, beta_rein=0.0,
            phi_s=2.5, phi_d=0.97,
            lambda_reinstate=0.0, tau_init=0.0, w_global=0.0,
            paradigm="free_recall",
        )
    else:
        # C&Z 2025 Table 1 (also serves as MS-TCM init; lambda/tau/w_global
        # seeded at non-trivial values for MS-TCM).
        p0 = ModelParameters(
            beta_enc=0.679, beta_enc_global=0.4, beta_story=0.400,
            gamma_fc=0.315, k=6.50, beta_rec=0.326, epsilon_d=1.04,
            beta_rein=0.300,
            lambda_reinstate=(0.5 if args.model == "mstcm" else 0.0),
            tau_init=(0.3 if args.model == "mstcm" else 0.0),
            w_global=(0.3 if args.model == "mstcm" else 0.0),
            paradigm="free_recall",
        )

    if args.model == "mstcm":
        params_to_theta = params_to_theta_mstcm
        theta_to_params = theta_to_params_mstcm
        n_theta = N_THETA_MSTCM
    elif args.model == "cmr":
        params_to_theta = params_to_theta_cmr
        theta_to_params = theta_to_params_cmr
        n_theta = N_THETA_CMR
    else:
        params_to_theta = params_to_theta_cz
        theta_to_params = theta_to_params_cz
        n_theta = N_THETA_CZ
    theta_init = params_to_theta(p0)

    print("  warming up...")
    initial_loss = loss(theta_init)
    print(f"  warmup done ({time.perf_counter()-t0:.1f}s); RMSE at init: {initial_loss:.6f}")

    rng = np.random.default_rng(args.seed)
    log_lines = [
        f"# Curve-matching fit ({args.model}) to frfr_category "
        f"(n_draws={args.n_draws}, n_restarts={args.n_restarts}, seed={args.seed})",
        f"RMSE at init = {initial_loss:.6f}",
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
        # The loss is stochastic (each eval simulates n_draws synthetic
        # datasets with the same master seed but different per-(draw,
        # part, lst) seeds). L-BFGS-B's default finite-difference step
        # eps ≈ 1.5e-8 is far below the loss's stochastic-noise floor,
        # so default-config optimizer declares "gradient = 0" at iter 0.
        # We use a much larger eps so the signal exceeds the noise.
        res = minimize(
            loss, theta0, method="L-BFGS-B",
            options={
                "maxiter": args.maxiter,
                "ftol": 1e-5,
                "gtol": 1e-4,
                "eps": 0.05,
            },
        )
        elapsed = time.perf_counter() - t_r
        log_lines.append(
            f"restart {r}: rmse0={loss(theta0):.6f}  rmse*={res.fun:.6f}  "
            f"converged={res.success}  iters={res.nit}  elapsed={elapsed:.1f}s",
        )
        print(log_lines[-1], flush=True)
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_theta = res.x.copy()
            best_restart = r

    total_t = time.perf_counter() - t_fit
    log_lines.append(
        f"best: rmse={best_loss:.6f}  restart={best_restart}  total fit time={total_t:.1f}s",
    )
    print(log_lines[-1])

    best_params = theta_to_params(best_theta)
    log_lines.append("")
    log_lines.append("MLE parameters (curve-matching):")
    cz_param_names = ("beta_enc", "beta_story", "gamma_fc", "k",
                      "beta_rec", "epsilon_d", "beta_rein")
    if args.model == "mstcm":
        param_names = cz_param_names + (
            "beta_enc_global", "lambda_reinstate", "tau_init", "w_global",
        )
    elif args.model == "cmr":
        param_names = ("beta_enc", "beta_rec", "gamma_fc", "k", "epsilon_d",
                       "phi_s", "phi_d")
    else:
        param_names = cz_param_names
    for name in param_names:
        log_lines.append(f"  {name} = {getattr(best_params, name):.4f}")

    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"\nLog written to: {log_path}")

    summary = {
        "model": {
            "mstcm": "mstcm_strict_hierarchical_free_recall_curves",
            "cz": "cz_hierarchical_free_recall_curves",
            "cmr": "polyn2009_standard_cmr_curves",
        }[args.model],
        "model_kind": args.model,
        "dataset": "frfr_category",
        "loss_type": "curve_rmse_equal_weight",
        "rmse": float(best_loss),
        "n_draws": args.n_draws,
        "n_restarts": args.n_restarts,
        "seed": args.seed,
        "n_lists": int(ds.presented.to_pandas()[["participant","list"]].drop_duplicates().shape[0]),
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
    summary_path = out_dir / "fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary written to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
