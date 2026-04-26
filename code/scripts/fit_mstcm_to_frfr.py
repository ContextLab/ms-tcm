"""Fit MS-TCM MS-TCM (strict hierarchical) to FRFR-category.

Mirrors `fit_mstcm_to_frfr.py` but uses MS-TCM core. Optimizes
10 parameters (beta_enc, beta_enc_global, beta_list, beta_rec,
beta_rein, gamma_fc, k, eps_d, lambda_reinstate, tau_init) to maximize
the marginalized + mixture log-likelihood.

Numpy-only fitter (no JAX autograd) because MS-TCM's LL has Python
control flow for visit-parsing that doesn't trace through jit. Uses
scipy L-BFGS-B with finite differences. Runtime: ~minutes per restart
on FRFR-category (480 lists, ~1k LL evals per fit).

Output:
    data/processed/fits/mstcm_frfr/fit_summary.json — MLE + LL.

Run from repo root:
    PYTHONPATH=code python code/scripts/fit_mstcm_to_frfr.py \\
        --n-restarts 3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from ms_tcm._likelihood_core_mstcm import (
    compute_list_log_likelihood_mstcm_numpy,
)
from ms_tcm.dataset import Dataset, load_dataset
from ms_tcm.frfr import load_frfr_category
from ms_tcm.params import ModelParameters


# 11 free parameters in optimizer-friendly unconstrained "theta" space:
#     theta[0]  = logit(beta_enc)
#     theta[1]  = logit(beta_list / beta_enc)        (enforces beta_list < beta_enc)
#     theta[2]  = logit(gamma_fc)
#     theta[3]  = log(k)
#     theta[4]  = logit(beta_rec)
#     theta[5]  = log(epsilon_d)
#     theta[6]  = logit(beta_rein)
#     theta[7]  = logit(lambda_reinstate)
#     theta[8]  = logit(tau_init)
#     theta[9]  = logit(beta_enc_global)
#     theta[10] = logit(w_global)


N_THETA = 11


def theta_to_params(theta) -> ModelParameters:
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
    lam = float(np.clip(expit(theta[7]), 0.0, 1.0))
    tau = float(np.clip(expit(theta[8]), 0.0, 1.0))
    beta_enc_g = float(np.clip(expit(theta[9]), eps, 1.0 - eps))
    w_global = float(np.clip(expit(theta[10]), 0.0, 1.0))
    return ModelParameters(
        beta_enc=beta_enc, beta_enc_global=beta_enc_g,
        beta_story=beta_list, gamma_fc=gamma_fc, k=k,
        beta_rec=beta_rec, epsilon_d=epsilon_d, beta_rein=beta_rein,
        lambda_reinstate=lam, tau_init=tau, w_global=w_global,
        paradigm="free_recall",
    )


def params_to_theta(p: ModelParameters) -> np.ndarray:
    eps = 1e-6
    theta = np.zeros(N_THETA, dtype=np.float64)
    theta[0] = logit(np.clip(p.beta_enc, eps, 1.0 - eps))
    theta[1] = logit(np.clip(p.beta_list / p.beta_enc, eps, 1.0 - eps))
    theta[2] = logit(np.clip(p.gamma_fc, eps, 1.0 - eps))
    theta[3] = np.log(p.k)
    theta[4] = logit(np.clip(p.beta_rec, eps, 1.0 - eps))
    theta[5] = np.log(p.epsilon_d)
    theta[6] = logit(np.clip(p.beta_rein, eps, 1.0 - eps))
    theta[7] = logit(np.clip(p.lambda_reinstate, eps, 1.0 - eps))
    theta[8] = logit(np.clip(p.tau_init, eps, 1.0 - eps))
    theta[9] = logit(np.clip(p.beta_enc_global, eps, 1.0 - eps))
    theta[10] = logit(np.clip(p.w_global, eps, 1.0 - eps))
    return theta


def build_per_list_inputs(ds: Dataset):
    """Return list of (W, K, cat_indices, recall_sps, recall_mask) tuples."""
    pdf = ds.presented.to_pandas()
    rdf = ds.recalled.to_pandas()
    keys = sorted(set(zip(pdf["participant"].tolist(), pdf["list"].tolist())))

    out = []
    for part, lst in keys:
        sub_p = pdf[(pdf["participant"] == part) & (pdf["list"] == lst)].sort_values("serial_position")
        sub_r = rdf[(rdf["participant"] == part) & (rdf["list"] == lst)].sort_values("output_position")

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

        sps = sub_r[sub_r["serial_position"] > 0]["serial_position"].to_numpy(dtype=np.int64)
        n = len(sps)
        R_pad = max(W, n, 1)
        recall_sps = np.zeros(R_pad, dtype=np.int64)
        recall_mask = np.zeros(R_pad, dtype=bool)
        recall_sps[:n] = sps
        recall_mask[:n] = True

        out.append((W, K, cat_indices, recall_sps, recall_mask))
    return out


def make_neg_loglik(ds: Dataset):
    """Return (nll_fn, n_lists) summing per-list LL across the dataset."""
    list_inputs = build_per_list_inputs(ds)
    n_lists = len(list_inputs)

    def nll(theta_np: np.ndarray) -> float:
        try:
            params = theta_to_params(theta_np)
        except Exception:
            return float("inf")
        total = 0.0
        for (W, K, cat_indices, recall_sps, recall_mask) in list_inputs:
            ll = compute_list_log_likelihood_mstcm_numpy(
                params, cat_indices, recall_sps, recall_mask, W=W, K=K,
            )
            if not np.isfinite(ll):
                return float("inf")
            total += ll
        return -total

    return nll, n_lists


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="frfr_category",
                        choices=["frfr_category", "kahana2002"])
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/processed/fits/mstcm_frfr")
    parser.add_argument("--restart-std", type=float, default=0.4)
    parser.add_argument("--maxiter", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = REPO / "notes/cz_reproduction/mstcm_fit_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "frfr_category":
        ds = load_frfr_category()
    else:
        ds = load_dataset(str(REPO / f"data/raw/{args.dataset}"))
    print(f"  {ds.num_participants} participants × {ds.num_lists_per_participant} lists × "
          f"{ds.num_words_per_list} words")

    print("Building negloglik (numpy, finite-difference gradients)...")
    t0 = time.perf_counter()
    nll, n_lists = make_neg_loglik(ds)
    print(f"  prepared {n_lists} lists for fit")

    # Initial parameters: similar to v1 fit but with reasonable defaults.
    p0 = ModelParameters(
        beta_enc=0.679, beta_enc_global=0.4, beta_story=0.400,
        gamma_fc=0.315, k=6.50, beta_rec=0.326, epsilon_d=1.04,
        beta_rein=0.300, lambda_reinstate=0.5, tau_init=0.3,
        w_global=0.3,
        paradigm="free_recall",
    )
    theta_init = params_to_theta(p0)

    print("  warming up by computing initial NLL...")
    initial_nll = nll(theta_init)
    print(f"  warmup done ({time.perf_counter()-t0:.1f}s); NLL at init: {initial_nll:.4f}")

    rng = np.random.default_rng(args.seed)
    log_lines = [
        f"# Fit MS-TCM MS-TCM to {args.dataset} "
        f"(n_restarts={args.n_restarts}, seed={args.seed})",
        f"NLL at init = {initial_nll:.4f}",
    ]

    best_nll = float("inf")
    best_theta = theta_init.copy()
    best_restart = -1

    t_fit = time.perf_counter()
    for r in range(args.n_restarts):
        if r == 0:
            theta0 = theta_init.copy()
        else:
            theta0 = theta_init + rng.normal(0, args.restart_std, size=N_THETA)
        t_r = time.perf_counter()
        res = minimize(
            nll, theta0, method="L-BFGS-B",
            options={"maxiter": args.maxiter, "ftol": 1e-7, "gtol": 1e-5},
        )
        elapsed = time.perf_counter() - t_r
        log_lines.append(
            f"restart {r}: nll0={nll(theta0):.4f}  nll*={res.fun:.4f}  "
            f"converged={res.success}  iters={res.nit}  elapsed={elapsed:.1f}s",
        )
        print(log_lines[-1], flush=True)
        if res.fun < best_nll:
            best_nll = float(res.fun)
            best_theta = res.x.copy()
            best_restart = r

    total_t = time.perf_counter() - t_fit
    log_lines.append(
        f"best: nll={best_nll:.4f}  ll={-best_nll:.4f}  restart={best_restart}  "
        f"total fit time={total_t:.1f}s",
    )
    print(log_lines[-1])

    best_params = theta_to_params(best_theta)
    log_lines.append("")
    log_lines.append("MLE parameters:")
    for name in ("beta_enc", "beta_enc_global", "beta_story", "gamma_fc", "k",
                 "beta_rec", "epsilon_d", "beta_rein",
                 "lambda_reinstate", "tau_init", "w_global"):
        log_lines.append(f"  {name} = {getattr(best_params, name):.4f}")

    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"\nLog written to: {log_path}")

    summary = {
        "model": "mstcm_strict_hierarchical_free_recall",
        "dataset": args.dataset,
        "log_likelihood": float(-best_nll),
        "negative_log_likelihood": float(best_nll),
        "n_restarts": args.n_restarts,
        "seed": args.seed,
        "n_lists": int(n_lists),
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
        },
        "theta_mle": [float(x) for x in best_theta],
    }
    summary_path = out_dir / "fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary written to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
