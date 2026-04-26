"""Fit Cornell & Zhang 2025 hierarchical free-recall model to FRFR-category.

Optimizes 7 parameters (β_enc, β_list, β_rec, β_rein, γ_fc, k, ε_d) to
maximize the marginalized log-likelihood under our new ``_likelihood_core``
(verified against equation-derived oracle to 1e-10, JAX-numpy parity to
machine epsilon). Uses JAX with analytic gradients for speed.

The free-recall variant of C&Z treats every list as a single storyline,
ignoring FRFR-category's category labels. β_rein is included for parity
with C&Z's full Table 1 but is unused in single-list FR.

Output:
    data/processed/fits/cz_frfr/fit_summary.json — MLE parameters + LL
    notes/cz_reproduction/fit_log.txt — restart-by-restart progress

Run from repo root:
    PYTHONPATH=code MS_TCM_JAX_DTYPE=float64 python code/scripts/fit_cz_to_frfr.py \\
        --n-restarts 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# JAX dtype must be set BEFORE importing jax.
os.environ.setdefault("MS_TCM_JAX_DTYPE", "float64")

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ms_tcm._likelihood_core import (
    CoreHyperparams,
    compute_list_log_likelihood,
)
from ms_tcm.dataset import Dataset, load_dataset
from ms_tcm.frfr import load_frfr_category
from ms_tcm.params import ModelParameters


# 7 free parameters in optimizer-friendly unconstrained "theta" space.
# Mapping:
#     theta[0] = logit(β_enc)
#     theta[1] = logit(β_list / β_enc)        # enforces β_list < β_enc (C&Z constraint)
#     theta[2] = logit(γ_fc)
#     theta[3] = log(k)
#     theta[4] = logit(β_rec)
#     theta[5] = log(ε_d)
#     theta[6] = logit(β_rein)


def theta_to_hyperparams_jax(theta):
    """Map (7,) theta -> CoreHyperparams (jax-tracers)."""
    from jax.scipy.special import expit as jexpit
    beta_enc = jexpit(theta[0])
    ratio = jexpit(theta[1])
    beta_list = beta_enc * ratio
    beta_list = jnp.minimum(beta_list, beta_enc - 1e-10)
    gamma_fc = jexpit(theta[2])
    k = jnp.exp(jnp.clip(theta[3], -30.0, 30.0))
    beta_rec = jexpit(theta[4])
    epsilon_d = jnp.exp(jnp.clip(theta[5], -30.0, 30.0))
    beta_rein = jexpit(theta[6])
    return CoreHyperparams(
        beta_enc=beta_enc, beta_list=beta_list, beta_rec=beta_rec,
        beta_rein=beta_rein, gamma_fc=gamma_fc, k=k, epsilon_d=epsilon_d,
    )


def theta_to_params_numpy(theta) -> ModelParameters:
    """Map (7,) theta -> ModelParameters (numpy floats), for human-readable summary."""
    beta_enc = float(expit(theta[0]))
    ratio = float(expit(theta[1]))
    beta_list = beta_enc * ratio
    beta_list = min(beta_list, beta_enc - 1e-10)
    gamma_fc = float(expit(theta[2]))
    k = float(np.exp(np.clip(theta[3], -30.0, 30.0)))
    beta_rec = float(expit(theta[4]))
    epsilon_d = float(np.exp(np.clip(theta[5], -30.0, 30.0)))
    beta_rein = float(expit(theta[6]))
    return ModelParameters(
        beta_enc=beta_enc, beta_story=beta_list, gamma_fc=gamma_fc, k=k,
        beta_rec=beta_rec, epsilon_d=epsilon_d, beta_rein=beta_rein,
        lambda_reinstate=0.0,  # not used by C&Z
        paradigm="free_recall",
    )


def params_to_theta_numpy(p: ModelParameters) -> np.ndarray:
    """Inverse of ``theta_to_params_numpy`` (clamped to a valid open interval)."""
    eps = 1e-6
    theta = np.zeros(7, dtype=np.float64)
    theta[0] = logit(np.clip(p.beta_enc, eps, 1.0 - eps))
    theta[1] = logit(np.clip(p.beta_list / p.beta_enc, eps, 1.0 - eps))
    theta[2] = logit(np.clip(p.gamma_fc, eps, 1.0 - eps))
    theta[3] = np.log(p.k)
    theta[4] = logit(np.clip(p.beta_rec, eps, 1.0 - eps))
    theta[5] = np.log(p.epsilon_d)
    theta[6] = logit(np.clip(p.beta_rein, eps, 1.0 - eps))
    return theta


def build_per_list_inputs(ds: Dataset):
    """Return (Ws, recall_sps_batch, recall_mask_batch) padded to global max R.

    For batched JAX evaluation, all lists must share the same R_pad. We
    pad every list's recall arrays to ``R_pad = max(W, n_recalls)`` across
    all lists. Intrusions (sp == 0) are dropped before padding.
    """
    pdf = ds.presented.to_pandas()
    rdf = ds.recalled.to_pandas()
    keys = sorted(set(zip(pdf["participant"].tolist(), pdf["list"].tolist())))

    Ws_list = []
    sps_list = []
    for part, lst in keys:
        sub_p = pdf[(pdf["participant"] == part) & (pdf["list"] == lst)]
        W = len(sub_p)
        sub_r = rdf[(rdf["participant"] == part) & (rdf["list"] == lst)].sort_values("output_position")
        sps = sub_r[sub_r["serial_position"] > 0]["serial_position"].to_numpy(dtype=np.int32)
        Ws_list.append(W)
        sps_list.append(sps)

    R_pad = max(max(W, len(sps)) for W, sps in zip(Ws_list, sps_list))
    n_lists = len(Ws_list)
    Ws = np.array(Ws_list, dtype=np.int32)
    recall_sps_batch = np.zeros((n_lists, R_pad), dtype=np.int32)
    recall_mask_batch = np.zeros((n_lists, R_pad), dtype=bool)
    for i, sps in enumerate(sps_list):
        n = len(sps)
        recall_sps_batch[i, :n] = sps
        recall_mask_batch[i, :n] = True

    # Sanity: all FRFR-category lists have W=16; the implementation
    # currently requires uniform W (the inner LL function is jit-compiled
    # for one specific W). If lists had varying W we'd need to bucket.
    Ws_unique = np.unique(Ws)
    assert len(Ws_unique) == 1, (
        f"non-uniform list lengths {Ws_unique.tolist()} not yet supported"
    )

    return int(Ws_unique[0]), recall_sps_batch, recall_mask_batch


def build_negloglik_with_grad(ds: Dataset):
    """Build (nll_fn, grad_fn) over 7-d theta using JAX autograd.

    Batches all lists into a single tensor and sums LL via ``jax.vmap``
    over the list axis. JIT compiles ONE function for the entire dataset
    (no Python-loop traces at the optimizer level).
    """
    W, recall_sps_batch, recall_mask_batch = build_per_list_inputs(ds)
    n_lists = recall_sps_batch.shape[0]
    R_pad = recall_sps_batch.shape[1]
    print(f"  prepared {n_lists} lists (W={W}, R_pad={R_pad}) for batched fit")

    sps_jax = jnp.asarray(recall_sps_batch, dtype=jnp.int32)
    mask_jax = jnp.asarray(recall_mask_batch, dtype=jnp.bool_)

    def _list_ll(theta, sps, mask):
        hp = theta_to_hyperparams_jax(theta)
        return compute_list_log_likelihood(
            hp, W=W,
            recall_sps=sps, recall_mask=mask,
            xp=jnp, dtype=jnp.float64,
        )

    # vmap over the list axis: in_axes=(None, 0, 0) means theta is shared,
    # sps & mask are batched along axis 0.
    _list_ll_v = jax.vmap(_list_ll, in_axes=(None, 0, 0))

    @jax.jit
    def _sum_ll(theta):
        return jnp.sum(_list_ll_v(theta, sps_jax, mask_jax))

    grad_sum_ll = jax.jit(jax.grad(_sum_ll))

    def nll(theta_np: np.ndarray) -> float:
        theta = jnp.asarray(theta_np, dtype=jnp.float64)
        return -float(_sum_ll(theta))

    def grad_nll(theta_np: np.ndarray) -> np.ndarray:
        theta = jnp.asarray(theta_np, dtype=jnp.float64)
        return -np.asarray(grad_sum_ll(theta), dtype=np.float64)

    return nll, grad_nll


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="frfr_category",
                        choices=["frfr_category", "kahana2002"])
    parser.add_argument("--n-restarts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/processed/fits/cz_frfr")
    parser.add_argument("--restart-std", type=float, default=0.4)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = REPO / "notes/cz_reproduction/fit_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "frfr_category":
        ds = load_frfr_category()
    else:
        ds = load_dataset(str(REPO / f"data/raw/{args.dataset}"))
    print(f"  {ds.num_participants} participants × {ds.num_lists_per_participant} lists × "
          f"{ds.num_words_per_list} words")

    print("Building negloglik + grad (JIT compile)...")
    t0 = time.perf_counter()
    nll, grad_nll = build_negloglik_with_grad(ds)
    # Warm up JIT.
    p0 = ModelParameters(
        beta_enc=0.679, beta_story=0.400, gamma_fc=0.315, k=6.50,
        beta_rec=0.326, epsilon_d=1.04, beta_rein=0.300,
        lambda_reinstate=0.0, paradigm="free_recall",
    )
    theta_init = params_to_theta_numpy(p0)
    print("  warming up JIT...")
    initial_nll = nll(theta_init)
    initial_grad = grad_nll(theta_init)
    print(f"  jit warmup ok ({time.perf_counter()-t0:.1f}s); "
          f"NLL at C&Z Table 1: {initial_nll:.4f}")

    rng = np.random.default_rng(args.seed)
    log_lines = [f"# Fit C&Z hierarchical to {args.dataset} (n_restarts={args.n_restarts}, seed={args.seed})"]
    log_lines.append(f"NLL at C&Z Table 1 init = {initial_nll:.4f}")

    best_nll = float("inf")
    best_theta = theta_init.copy()
    best_restart = -1

    t_fit = time.perf_counter()
    for r in range(args.n_restarts):
        if r == 0:
            theta0 = theta_init.copy()
        else:
            theta0 = theta_init + rng.normal(0, args.restart_std, size=7)
        t_r = time.perf_counter()
        res = minimize(nll, theta0, jac=grad_nll, method="L-BFGS-B",
                       options={"maxiter": 200})
        elapsed = time.perf_counter() - t_r
        log_lines.append(
            f"restart {r}: nll0={nll(theta0):.4f}  nll*={res.fun:.4f}  "
            f"converged={res.success}  iters={res.nit}  elapsed={elapsed:.1f}s",
        )
        print(log_lines[-1])
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

    best_params = theta_to_params_numpy(best_theta)
    log_lines.append("")
    log_lines.append("MLE parameters:")
    for name in ("beta_enc", "beta_story", "gamma_fc", "k",
                 "beta_rec", "epsilon_d", "beta_rein"):
        val = getattr(best_params, name)
        log_lines.append(f"  {name} = {val:.4f}")
    log_lines.append("(beta_list = beta_story; identical numerical value, C&Z naming)")

    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"\nLog written to: {log_path}")

    # Save fit_summary.json so figures can pick it up.
    summary = {
        "model": "cz_hierarchical_free_recall",
        "dataset": args.dataset,
        "log_likelihood": float(-best_nll),
        "negative_log_likelihood": float(best_nll),
        "n_restarts": args.n_restarts,
        "seed": args.seed,
        "n_lists": int(ds.presented.to_pandas()[["participant","list"]].drop_duplicates().shape[0]),
        "elapsed_seconds": float(total_t),
        "parameters": {
            "beta_enc": float(best_params.beta_enc),
            "beta_list": float(best_params.beta_list),
            "beta_story": float(best_params.beta_story),  # alias
            "gamma_fc": float(best_params.gamma_fc),
            "k": float(best_params.k),
            "beta_rec": float(best_params.beta_rec),
            "epsilon_d": float(best_params.epsilon_d),
            "beta_rein": float(best_params.beta_rein),
        },
        "theta_mle": [float(x) for x in best_theta],
    }
    summary_path = out_dir / "fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary written to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
