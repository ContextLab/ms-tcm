"""Equation-derived oracle tests for MS-TCM (`_likelihood_core_mstcm`).

Mirrors the C&Z oracle pattern in `test_likelihood_core.py`. Two tests:

1. **Reduction-to-C&Z**: with K=1 and τ=0, MS-TCM must produce identical
   LLs to C&Z's `compute_list_log_likelihood_numpy` (within float
   rounding). This anchors MS-TCM correctness against the already-verified
   C&Z implementation.

2. **Naive oracle**: an independent line-by-line implementation of MS-TCM
   (encoding + retrieval + initiation mixture) using only Python floats /
   numpy. The core's outputs must match this oracle to 1e-10 across
   random small lists with K storylines.
"""

from __future__ import annotations

import numpy as np
import pytest

from ms_tcm._likelihood_core import compute_list_log_likelihood_numpy
from ms_tcm._likelihood_core_mstcm import (
    compute_list_log_likelihood_mstcm_numpy,
)
from ms_tcm.params import ModelParameters


# -----------------------------------------------------------------------
# Naive oracle: line-by-line implementation of MS-TCM equations.
# -----------------------------------------------------------------------


def _onehot(d, i):
    v = np.zeros(d, dtype=np.float64)
    v[i] = 1.0
    return v


def _rho(beta, dot):
    inner = 1.0 + beta * beta * (dot * dot - 1.0)
    return np.sqrt(max(0.0, inner)) - beta * dot


def _drift(c_prev, c_in, beta):
    dot = float(np.dot(c_prev, c_in))
    rho = _rho(beta, dot)
    return rho * c_prev + beta * c_in


def _log_softmax_masked(scaled, keep_mask):
    s = np.where(keep_mask, scaled, -1.0e30)
    m = s.max()
    shifted = s - m
    exp_s = np.where(keep_mask, np.exp(shifted), 0.0)
    denom = max(exp_s.sum(), 1e-300)
    return s - m - np.log(denom)


def _p_stop(m_cf_exp, c_ret, already, eps_d):
    a = m_cf_exp @ c_ret
    a_abs = np.abs(a)
    a_r = float(np.sum(np.where(already, a_abs, 0.0)))
    a_nr = float(np.sum(np.where(already, 0.0, a_abs)))
    if a_r <= 0:
        return 0.0
    return float(np.exp(-eps_d * a_nr / a_r))


def _c_IN_rec(idx, m_fc_exp, gamma_fc, d):
    pre = _onehot(d, idx + 1)
    return (1.0 - gamma_fc) * pre + gamma_fc * m_fc_exp[:, idx]


def _ll_for_seed(c_ret_init, m_fc_exp, m_cf_exp, p, recall_sps, recall_mask, W):
    """C&Z marginalized LL given an arbitrary initial c_ret seed.

    Mirrors the marginalization in C&Z's compute_list_log_likelihood —
    factored to allow MS-TCM to invoke it with different seeds.
    """
    d = W + 1
    beta_rec = float(p.beta_rec)
    gamma_fc = float(p.gamma_fc)
    k = float(p.k)
    eps_d = float(p.epsilon_d)
    R_total = recall_sps.shape[0]
    e_start = _onehot(d, 0)

    # Build the observed already_mask trajectory (independent of T;
    # repeats are noise — skip mask update).
    mask_traj = [np.zeros(W, dtype=bool)]
    cur_mask = np.zeros(W, dtype=bool)
    for i in range(R_total):
        if bool(recall_mask[i]):
            sp = int(recall_sps[i])
            idx = max(0, min(W - 1, sp - 1))
            if not cur_mask[idx]:
                cur_mask = cur_mask.copy()
                cur_mask[idx] = True
        mask_traj.append(cur_mask.copy())

    # Phase-1 forward (cued by c_ret_init).
    c_ret_p1_traj = [c_ret_init.copy()]
    cum_p1 = [0.0]
    cum = 0.0
    c_ret = c_ret_init.copy()
    for i in range(R_total):
        valid = bool(recall_mask[i])
        sp = int(recall_sps[i])
        idx = max(0, min(W - 1, sp - 1))
        already = mask_traj[i]
        is_repeat = bool(already[idx])
        countable = valid and not is_repeat
        if countable:
            ps = _p_stop(m_cf_exp, c_ret, already, eps_d)
            log_1m = float(np.log(max(1.0 - ps, 1e-300)))
            a = m_cf_exp @ c_ret
            log_p = _log_softmax_masked(k * a, ~already)
            cum += log_1m + float(log_p[idx])
            c_in = _c_IN_rec(idx, m_fc_exp, gamma_fc, d)
            c_ret = _drift(c_ret, c_in, beta_rec)
        cum_p1.append(cum)
        c_ret_p1_traj.append(c_ret.copy())

    # Per-T marginalization.
    ll_per_T = np.zeros(R_total + 1, dtype=np.float64)
    for T in range(R_total + 1):
        phase1_ll = cum_p1[T]
        c_ret_at_trans = c_ret_p1_traj[T]
        mask_at_trans = mask_traj[T]
        ps_trans = _p_stop(m_cf_exp, c_ret_at_trans, mask_at_trans, eps_d)
        log_trans = float(np.log(max(ps_trans, 1e-300)))

        # Phase-2 forward from slot T (cued by e_start).
        c_ret_2 = e_start.copy()
        mask_2 = mask_at_trans.copy()
        phase2_ll = 0.0
        for i in range(T, R_total):
            valid = bool(recall_mask[i])
            sp = int(recall_sps[i])
            idx = max(0, min(W - 1, sp - 1))
            is_repeat = bool(mask_2[idx])
            countable = valid and not is_repeat
            if countable:
                ps_i = _p_stop(m_cf_exp, c_ret_2, mask_2, eps_d)
                log_1m = float(np.log(max(1.0 - ps_i, 1e-300)))
                a = m_cf_exp @ c_ret_2
                log_p = _log_softmax_masked(k * a, ~mask_2)
                phase2_ll += log_1m + float(log_p[idx])
                c_in = _c_IN_rec(idx, m_fc_exp, gamma_fc, d)
                c_ret_2 = _drift(c_ret_2, c_in, beta_rec)
                mask_2 = mask_2.copy()
                mask_2[idx] = True

        ps_final = _p_stop(m_cf_exp, c_ret_2, mask_2, eps_d)
        log_final = float(np.log(max(ps_final, 1e-300)))
        ll_per_T[T] = phase1_ll + log_trans + phase2_ll + log_final

    m = ll_per_T.max()
    return float(m + np.log(np.sum(np.exp(ll_per_T - m))))


def _oracle_mstcm_ll(
    p: ModelParameters,
    W: int,
    K: int,
    cat_indices: np.ndarray,
    recall_sps: np.ndarray,
    recall_mask: np.ndarray,
) -> float:
    """Independent oracle for MS-TCM list LL.

    Encoding: per-storyline drift + λ reinstatement + M^SC accumulation.
    Retrieval: τ-mixture between recency-init and storyline-init routes.
    """
    d = W + 1
    beta_enc = float(p.beta_enc)
    beta_list = float(p.beta_list)
    gamma_fc = float(p.gamma_fc)
    k = float(p.k)
    lam = float(p.lambda_reinstate)
    tau = float(p.tau_init)

    # --- Encoding ---
    c_item = _onehot(d, 0)
    c_list = _onehot(d, 0)
    c_story = np.tile(_onehot(d, 0)[None, :], (K, 1))
    m_fc_exp = np.zeros((d, W), dtype=np.float64)
    m_cf_exp = np.zeros((W, d), dtype=np.float64)
    m_sc = np.zeros((K, d), dtype=np.float64)
    seen = np.zeros(K, dtype=bool)
    prev_s = -1

    c_item_traj = [c_item.copy()]

    for t in range(W):
        s = int(cat_indices[t])
        m_fc_exp[:, t] += c_item
        m_cf_exp[t, :] += c_item
        c_in = _onehot(d, t + 1)
        c_item = _drift(c_item, c_in, beta_enc)
        c_list = _drift(c_list, c_in, beta_list)

        is_switch = (prev_s >= 0) and (prev_s != s)
        is_return = is_switch and bool(seen[s])

        if is_switch:
            m_sc[prev_s, :] = c_story[prev_s, :]

        if is_return:
            cached = m_sc[s, :].copy()
            current = c_story[s, :].copy()
            blended = lam * cached + (1.0 - lam) * current
            n = float(np.linalg.norm(blended))
            if n > 1e-12:
                blended = blended / n
            c_story[s, :] = blended

        c_story_active = c_story[s, :].copy()
        c_story[s, :] = _drift(c_story_active, c_in, beta_list)

        seen[s] = True
        prev_s = s
        c_item_traj.append(c_item.copy())

    # Final cache update for last active storyline.
    if prev_s >= 0:
        m_sc[prev_s, :] = c_story[prev_s, :]

    c_item_end = c_item_traj[W]
    c_list_end = c_list

    # --- Compute storyline-selection log-probabilities ---
    storyline_acts = m_sc @ c_list_end
    log_p_s = _log_softmax_masked(k * storyline_acts, np.ones(K, dtype=bool))

    # --- LL per route ---
    ll_recency = _ll_for_seed(
        c_item_end, m_fc_exp, m_cf_exp, p, recall_sps, recall_mask, W,
    )
    ll_storylines = np.array(
        [_ll_for_seed(c_story[s, :], m_fc_exp, m_cf_exp, p,
                       recall_sps, recall_mask, W)
         for s in range(K)],
        dtype=np.float64,
    )
    combined = log_p_s + ll_storylines
    m = combined.max()
    ll_storyline_route = m + np.log(np.sum(np.exp(combined - m)))

    # Mix.
    log_tau = np.log(max(tau, 1e-300))
    log_1m_tau = np.log(max(1.0 - tau, 1e-300))
    a_term = log_1m_tau + ll_recency
    b_term = log_tau + ll_storyline_route
    m = max(a_term, b_term)
    return float(m + np.log(np.exp(a_term - m) + np.exp(b_term - m)))


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


def _default_params(lambda_reinstate=0.5, tau_init=0.5):
    return ModelParameters(
        beta_enc=0.679, beta_story=0.400, gamma_fc=0.315, k=6.50,
        beta_rec=0.326, epsilon_d=1.04, beta_rein=0.300,
        lambda_reinstate=lambda_reinstate, tau_init=tau_init,
        paradigm="free_recall",
    )


def _random_recalls(rng, W, max_R=None):
    if max_R is None:
        max_R = W
    R = int(rng.integers(1, max_R + 1))
    sps = (rng.permutation(W)[:R] + 1).tolist()
    R_pad = max(R, 1)
    recall_sps = np.zeros(R_pad, dtype=np.int64)
    recall_mask = np.zeros(R_pad, dtype=bool)
    recall_sps[:R] = sps
    recall_mask[:R] = True
    return recall_sps, recall_mask


def _random_cat_indices(rng, W, K):
    """K storylines, each occupying W/K consecutive items (blocked)."""
    if K == 1:
        return np.zeros(W, dtype=np.int64)
    return np.repeat(np.arange(K, dtype=np.int64), W // K)[:W]


@pytest.mark.parametrize("seed", list(range(5)))
def test_mstcm_reduces_to_cz_when_K1_and_tau0(seed):
    """K=1, τ=0: MS-TCM ≡ C&Z to machine epsilon."""
    rng = np.random.default_rng(seed)
    W = 8
    p = _default_params(lambda_reinstate=0.0, tau_init=0.0)
    cat_indices = np.zeros(W, dtype=np.int64)
    recall_sps, recall_mask = _random_recalls(rng, W)
    ll_mstcm = compute_list_log_likelihood_mstcm_numpy(
        p, cat_indices, recall_sps, recall_mask, W=W, K=1,
    )
    ll_cz = compute_list_log_likelihood_numpy(p, recall_sps, recall_mask, W=W)
    assert abs(ll_mstcm - ll_cz) < 1e-10, (
        f"MS-TCM (K=1, τ=0) should equal C&Z; got mstcm={ll_mstcm!r}, "
        f"cz={ll_cz!r}, diff={ll_mstcm - ll_cz!r}"
    )


@pytest.mark.parametrize("seed", list(range(5)))
@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("tau,lam", [
    (0.0, 0.0),
    (0.5, 0.5),
    (0.9, 0.0),
    (0.3, 0.8),
])
def test_mstcm_matches_oracle(seed, K, tau, lam):
    """MS-TCM core LL matches naive oracle to 1e-10."""
    rng = np.random.default_rng(seed * 13 + K)
    W = 8 if K == 2 else 12
    p = _default_params(lambda_reinstate=lam, tau_init=tau)
    cat_indices = _random_cat_indices(rng, W, K)
    recall_sps, recall_mask = _random_recalls(rng, W)
    ll_oracle = _oracle_mstcm_ll(
        p, W, K, cat_indices, recall_sps, recall_mask,
    )
    ll_core = compute_list_log_likelihood_mstcm_numpy(
        p, cat_indices, recall_sps, recall_mask, W=W, K=K,
    )
    assert np.isclose(ll_oracle, ll_core, atol=1e-10, rtol=0), (
        f"core diverges from oracle: oracle={ll_oracle!r}, core={ll_core!r}, "
        f"diff={ll_oracle - ll_core!r}\n"
        f"  W={W} K={K} τ={tau} λ={lam} cat={cat_indices.tolist()} "
        f"recalls={recall_sps.tolist()}"
    )


def test_mstcm_storyline_routes_increase_pFR_at_storyline_starts():
    """Sanity: with τ>0, simulated pFR puts mass at storyline-start positions."""
    from ms_tcm._likelihood_core_mstcm import simulate_recalls_mstcm
    W = 16
    K = 4
    cat_indices = np.repeat(np.arange(K), W // K)  # blocked
    p_no_tau = _default_params(lambda_reinstate=0.0, tau_init=0.0)
    p_with_tau = _default_params(lambda_reinstate=0.5, tau_init=0.7)

    first_recalls_tau0 = []
    first_recalls_tau07 = []
    for trial in range(200):
        rng = np.random.default_rng(trial)
        rec_a = simulate_recalls_mstcm(
            p_no_tau, W=W, K=K, cat_indices=cat_indices, rng=rng,
        )
        if rec_a:
            first_recalls_tau0.append(rec_a[0])
        rng = np.random.default_rng(trial + 10000)
        rec_b = simulate_recalls_mstcm(
            p_with_tau, W=W, K=K, cat_indices=cat_indices, rng=rng,
        )
        if rec_b:
            first_recalls_tau07.append(rec_b[0])

    frac_pos1_tau0 = sum(1 for s in first_recalls_tau0 if s == 1) / len(first_recalls_tau0)
    frac_pos1_tau07 = sum(1 for s in first_recalls_tau07 if s == 1) / len(first_recalls_tau07)
    # τ=0: pFR(sp=1) should be very small (recency-dominant). τ=0.7 should
    # produce a substantial fraction. Threshold loosely.
    assert frac_pos1_tau0 < 0.10, (
        f"τ=0 should give small pFR(sp=1); got {frac_pos1_tau0:.3f}"
    )
    assert frac_pos1_tau07 > 0.10, (
        f"τ=0.7 should give substantial pFR(sp=1); got {frac_pos1_tau07:.3f}"
    )
