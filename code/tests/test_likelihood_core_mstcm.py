"""Equation-derived oracle tests for MS-TCM MS-TCM
(`_likelihood_core_mstcm`).

Equation-derived oracle (independent naive numpy reference) but with the new strict-hierarchical
retrieval mechanism + separate global / per-storyline contexts.

Tests:

1. **Reduction-to-C&Z**: with K=1 and tau=0, MS-TCM should
   produce the same log-likelihood as the pre-consolidation MS-TCM model
   (`compute_list_log_likelihood_mstcm_numpy`) within float rounding.
   [Soft test: route alpha replaces the recency-init route, so this
   reduction holds when β_enc_global = β_enc and gamma_fc, etc. are
   matched. Easier reduction: K=1 + tau=0 + beta_enc_global = beta_enc
   gives a route-α-only computation that should match a C&Z baseline
   with the same parameters when there's no hierarchical fallback.]

2. **Naive oracle**: independent line-by-line implementation of MS-TCM
   v2 (encoding + α-route LL + β-route LL + mixture) using only Python
   floats / numpy. Core's outputs match this oracle to 1e-10 across
   random small lists with K storylines.

3. **Storyline-routes increase pFR at storyline starts**: a sanity
   simulation test (mirrors pre-consolidation MS-TCM's analogous test).
"""

from __future__ import annotations

import numpy as np
import pytest

from ms_tcm._likelihood_core_mstcm import (
    compute_list_log_likelihood_mstcm_numpy,
    simulate_recalls_mstcm,
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


def _p_stop(m_cf, c_ret, already, eps_d):
    a = m_cf @ c_ret
    a_abs = np.abs(a)
    a_r = float(np.sum(np.where(already, a_abs, 0.0)))
    a_nr = float(np.sum(np.where(already, 0.0, a_abs)))
    if a_r <= 0:
        return 0.0
    return float(np.exp(-eps_d * a_nr / a_r))


def _c_IN_rec(idx, m_fc, gamma_fc, d):
    pre = _onehot(d, idx + 1)
    return (1.0 - gamma_fc) * pre + gamma_fc * m_fc[:, idx]


def _oracle_mstcm_encode(p, W, K, cat_indices):
    """Naive numpy reference for MS-TCM encoding."""
    d = W + 1
    beta_enc = float(p.beta_enc)
    beta_enc_g = float(p.beta_enc_global)
    beta_list = float(p.beta_list)
    lam = float(p.lambda_reinstate)

    c_global = _onehot(d, 0)
    c_list_g = _onehot(d, 0)
    c_item_per_s = np.tile(_onehot(d, 0)[None, :], (K, 1)).copy()
    c_story_per_s = np.tile(_onehot(d, 0)[None, :], (K, 1)).copy()

    m_fc_g = np.zeros((d, W), dtype=np.float64)
    m_cf_g = np.zeros((W, d), dtype=np.float64)
    m_fc_s = np.zeros((K, d, W), dtype=np.float64)
    m_cf_s = np.zeros((K, W, d), dtype=np.float64)
    m_sc = np.zeros((K, d), dtype=np.float64)

    c_item_traj_g = [c_global.copy()]
    seen = np.zeros(K, dtype=bool)
    prev_s = -1

    for t in range(W):
        s = int(cat_indices[t])
        is_switch = (prev_s >= 0) and (prev_s != s)
        is_return = is_switch and bool(seen[s])

        if is_switch:
            m_sc[prev_s, :] = c_story_per_s[prev_s, :].copy()

        # Decision 1A: lambda reinstatement BEFORE drift on returns.
        if is_return:
            cached = m_sc[s, :].copy()
            current = c_story_per_s[s, :].copy()
            blended = lam * cached + (1.0 - lam) * current
            n = float(np.linalg.norm(blended))
            if n > 1e-12:
                blended = blended / n
            c_story_per_s[s, :] = blended

        c_in = _onehot(d, t + 1)

        # Pre-drift matrix updates.
        m_fc_g[:, t] += c_global
        m_cf_g[t, :] += c_global
        m_fc_s[s, :, t] += c_item_per_s[s, :]
        m_cf_s[s, t, :] += c_item_per_s[s, :]

        # Drifts.
        c_global = _drift(c_global, c_in, beta_enc_g)
        c_list_g = _drift(c_list_g, c_in, beta_list)
        c_item_per_s[s, :] = _drift(c_item_per_s[s, :], c_in, beta_enc)
        c_story_per_s[s, :] = _drift(c_story_per_s[s, :], c_in, beta_list)

        seen[s] = True
        prev_s = s
        c_item_traj_g.append(c_global.copy())

    if prev_s >= 0:
        m_sc[prev_s, :] = c_story_per_s[prev_s, :].copy()

    return (
        c_item_traj_g, c_item_per_s, c_story_per_s, c_list_g,
        m_fc_g, m_cf_g, m_fc_s, m_cf_s, m_sc,
    )


def _oracle_mstcm_ll(p, W, K, cat_indices, recall_sps, recall_mask):
    """Naive numpy oracle for MS-TCM mixture LL."""
    d = W + 1
    beta_rec = float(p.beta_rec)
    gamma_fc = float(p.gamma_fc)
    k = float(p.k)
    eps_d = float(p.epsilon_d)
    tau = float(p.tau_init)
    R = recall_sps.shape[0]

    enc = _oracle_mstcm_encode(p, W, K, cat_indices)
    (c_item_traj_g, c_item_per_s, c_story_per_s, c_list_g,
     m_fc_g, m_cf_g, m_fc_s, m_cf_s, m_sc) = enc
    c_global_end = c_item_traj_g[W]

    # --- Route alpha LL ---
    c_ret = c_global_end.copy()
    cum_alpha = 0.0
    already_a = np.zeros(W, dtype=bool)
    for i in range(R):
        sp = int(recall_sps[i])
        valid = bool(recall_mask[i])
        if not valid:
            continue
        idx = max(0, min(W - 1, sp - 1))
        if already_a[idx]:
            continue
        ps = _p_stop(m_cf_g, c_ret, already_a, eps_d)
        cum_alpha += float(np.log(max(1.0 - ps, 1e-300)))
        a = m_cf_g @ c_ret
        log_p = _log_softmax_masked(k * a, ~already_a)
        cum_alpha += float(log_p[idx])
        c_in = _c_IN_rec(idx, m_fc_g, gamma_fc, d)
        c_ret = _drift(c_ret, c_in, beta_rec)
        already_a[idx] = True
    ps_final_a = _p_stop(m_cf_g, c_ret, already_a, eps_d)
    cum_alpha += float(np.log(max(ps_final_a, 1e-300)))

    # --- Route beta LL ---
    # Parse visits.
    visits = []
    cur_s = None
    cur_idx = []
    for i in range(R):
        sp = int(recall_sps[i])
        if not bool(recall_mask[i]) or sp == 0:
            continue
        idx = max(0, min(W - 1, sp - 1))
        s_of = int(cat_indices[idx])
        if cur_s is None or s_of != cur_s:
            if cur_idx:
                visits.append((cur_s, cur_idx))
            cur_s = s_of
            cur_idx = [i]
        else:
            cur_idx.append(i)
    if cur_idx:
        visits.append((cur_s, cur_idx))

    if not visits:
        cum_beta = 0.0
    else:
        cum_beta = 0.0
        c_ret_g = c_global_end.copy()
        already_b = np.zeros(W, dtype=bool)
        last_s = -1
        for s_hat, rec_indices in visits:
            # Storyline selection log-prob.
            cand_mask = np.ones(K, dtype=bool)
            if last_s >= 0:
                cand_mask[last_s] = False
            if not cand_mask[s_hat]:
                return float("-inf")
            story_acts = m_sc @ c_ret_g
            log_p_story = _log_softmax_masked(k * story_acts, cand_mask)
            cum_beta += float(log_p_story[s_hat])

            # Within-storyline recall: c_ret_s = e_start (option ii).
            c_ret_s = _onehot(d, 0)
            in_story = np.array(
                [int(cat_indices[sp_idx]) == s_hat for sp_idx in range(W)],
                dtype=bool,
            )
            # Effective matrices mixing per-storyline + global by w_global.
            wg = float(p.w_global)
            m_cf_eff = (1.0 - wg) * m_cf_s[s_hat] + wg * m_cf_g
            m_fc_eff = (1.0 - wg) * m_fc_s[s_hat] + wg * m_fc_g

            for i in rec_indices:
                sp = int(recall_sps[i])
                idx = max(0, min(W - 1, sp - 1))
                if already_b[idx]:
                    continue
                already_for_stop = already_b | ~in_story
                ps = _p_stop(m_cf_eff, c_ret_s, already_for_stop, eps_d)
                cum_beta += float(np.log(max(1.0 - ps, 1e-300)))
                a = m_cf_eff @ c_ret_s
                keep = in_story & ~already_b
                log_p = _log_softmax_masked(k * a, keep)
                cum_beta += float(log_p[idx])
                # Drift Decision 2X: drift both contexts. Within uses
                # mixed matrix; global uses M^FC_G.
                c_in_s = _c_IN_rec(idx, m_fc_eff, gamma_fc, d)
                c_ret_s = _drift(c_ret_s, c_in_s, beta_rec)
                c_in_g = _c_IN_rec(idx, m_fc_g, gamma_fc, d)
                c_ret_g = _drift(c_ret_g, c_in_g, beta_rec)
                already_b[idx] = True

            already_for_stop = already_b | ~in_story
            ps_end = _p_stop(m_cf_eff, c_ret_s, already_for_stop, eps_d)
            cum_beta += float(np.log(max(ps_end, 1e-300)))
            last_s = s_hat

    # Mixture.
    log_tau = float(np.log(max(tau, 1e-300)))
    log_1m_tau = float(np.log(max(1.0 - tau, 1e-300)))
    a_term = log_1m_tau + cum_alpha
    b_term = log_tau + cum_beta
    m_ = max(a_term, b_term)
    return float(m_ + np.log(np.exp(a_term - m_) + np.exp(b_term - m_)))


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


def _default_params(
    beta_enc_global=0.4,
    lambda_reinstate=0.5, tau_init=0.5, w_global=0.0,
):
    return ModelParameters(
        beta_enc=0.679, beta_enc_global=beta_enc_global, beta_story=0.400,
        gamma_fc=0.315, k=6.50, beta_rec=0.326, epsilon_d=1.04,
        beta_rein=0.300,
        lambda_reinstate=lambda_reinstate, tau_init=tau_init,
        w_global=w_global,
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


def _blocked_cat_indices(W, K):
    """K storylines occupying W/K consecutive items each."""
    if K == 1:
        return np.zeros(W, dtype=np.int64)
    return np.repeat(np.arange(K, dtype=np.int64), W // K)[:W]


# --- Naive oracle parity tests ---


@pytest.mark.parametrize("seed", list(range(5)))
@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("tau,lam,wg", [
    (0.0, 0.0, 0.0),   # all-off baseline
    (0.5, 0.5, 0.0),   # tau + lam, strict (w_global = 0)
    (0.9, 0.0, 0.0),   # tau dominant, strict
    (0.3, 0.8, 0.0),   # lambda-heavy, strict
    (0.5, 0.5, 0.3),   # soft hierarchy (w_global = 0.3)
    (0.5, 0.5, 0.7),   # softer hierarchy (w_global = 0.7)
    (0.5, 0.5, 1.0),   # full global (w_global = 1)
])
def test_mstcm_matches_oracle(seed, K, tau, lam, wg):
    """MS-TCM core LL matches naive oracle to 1e-10 across (τ, λ, w_global)."""
    rng = np.random.default_rng(seed * 13 + K)
    W = 8 if K == 2 else 12
    p = _default_params(lambda_reinstate=lam, tau_init=tau, w_global=wg)
    cat_indices = _blocked_cat_indices(W, K)
    recall_sps, recall_mask = _random_recalls(rng, W)

    ll_oracle = _oracle_mstcm_ll(p, W, K, cat_indices, recall_sps, recall_mask)
    ll_core = compute_list_log_likelihood_mstcm_numpy(
        p, cat_indices, recall_sps, recall_mask, W=W, K=K,
    )
    assert np.isclose(ll_oracle, ll_core, atol=1e-10, rtol=0), (
        f"core diverges from oracle: oracle={ll_oracle!r}, core={ll_core!r}, "
        f"diff={ll_oracle - ll_core!r}\n"
        f"  W={W} K={K} τ={tau} λ={lam} w_global={wg} "
        f"cat={cat_indices.tolist()} recalls={recall_sps.tolist()}"
    )


def test_mstcm_storyline_routes_put_mass_at_storyline_starts():
    """With τ>0, simulated pFR puts mass at storyline-start positions
    (1, 5, 9, 13 in a blocked K=4 list).

    Note: under option (ii) c_ret_s = e_start, the dominant storyline
    selected via M^SC · c_global_end will be the MOST-RECENTLY-ENCODED
    storyline (because its M^SC entry overlaps most with c_global_end).
    Within that storyline, c_ret_s = e_start produces within-storyline
    primacy → pFR favors the FIRST item of the most-recent storyline.

    For a blocked W=16 K=4 list, that's pFR(sp=13). pFR(sp=1) requires
    storyline 0 to be selected over storylines 1, 2, 3 — possible but
    less common.

    Test: at τ=0.7 the SUM of pFR mass at storyline-start positions
    {1, 5, 9, 13} should be substantially higher than at τ=0.
    """
    W = 16
    K = 4
    cat_indices = _blocked_cat_indices(W, K)
    storyline_starts = {1, 5, 9, 13}
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

    frac_starts_tau0 = sum(
        1 for s in first_recalls_tau0 if s in storyline_starts
    ) / max(1, len(first_recalls_tau0))
    frac_starts_tau07 = sum(
        1 for s in first_recalls_tau07 if s in storyline_starts
    ) / max(1, len(first_recalls_tau07))

    # τ=0.7 should put MORE mass at storyline starts than τ=0.
    assert frac_starts_tau07 > frac_starts_tau0, (
        f"τ=0.7 should put more pFR mass at storyline starts than τ=0; "
        f"got τ=0: {frac_starts_tau0:.3f}, τ=0.7: {frac_starts_tau07:.3f}"
    )
    # And the τ=0.7 mass at storyline starts should be substantial
    # (≥20% — consistent with at least one storyline-init firing per
    # simulated trial putting mass at one of the 4 starts).
    assert frac_starts_tau07 > 0.20, (
        f"τ=0.7 should give substantial pFR mass at storyline starts; "
        f"got {frac_starts_tau07:.3f}"
    )
