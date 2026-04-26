"""Equation-derived oracle tests for ``ms_tcm._likelihood_core`` (C&Z 2025).

This file is the correctness anchor for Constitution II (Single Source of
Truth). It contains an INDEPENDENT, naive numpy implementation of Cornell
& Zhang 2025's hierarchical free-recall model — line-by-line from the
equations in the paper, with no optimization, no shared helpers, and no
calls into ``_likelihood_core``. The core's outputs must match this oracle
to within machine epsilon on every test case.

Equations implemented (C&Z 2025, free-recall variant):
    Eq 1  : c^item_i = ρ c^item_{i-1} + β_enc c^IN_enc
    Eq 2a : ΔM^FC_exp = c^item_{i-1} f_i^T
    Eq 2b : ΔM^CF_exp = f_i c^item_{i-1}^T
    Eq 3  : c^ret_j   = ρ c^ret_{j-1} + β_rec c^IN_rec
    Eq 4  : c^IN_rec  = (1-γ_fc) M^FC_pre · f_j + γ_fc M^FC_exp · f_j
    Eq 5  : a_j       = M^CF_exp · c^ret_j   (NO primacy gradient)
    Eq 6  : p(j)      = softmax(k · a_j)     (with mask over already-recalled)
    Eq 7  : p_stop    = exp(-ε_d · a^nr / a^r)
    Eq 8  : c^list_i  = ρ c^list_{i-1} + β_list c^IN_enc
    Eq 15 : reinstate c^item ← e_start (beginning-of-list) on phase-1 stop

Marginalized over phase-transition latent T ∈ {0, ..., R}.
"""

from __future__ import annotations

import numpy as np
import pytest

from ms_tcm._likelihood_core import compute_list_log_likelihood_numpy
from ms_tcm.params import ModelParameters


# -----------------------------------------------------------------------
# Oracle: naive line-by-line implementation of C&Z 2025 equations.
# -----------------------------------------------------------------------


def _onehot(d, i):
    v = np.zeros(d, dtype=np.float64)
    v[i] = 1.0
    return v


def _rho(beta, dot):
    inner = 1.0 + beta * beta * (dot * dot - 1.0)
    return np.sqrt(max(0.0, inner)) - beta * dot


def _drift(c_prev, c_in, beta):
    """C&Z Eqs 1, 3, 8: c <- ρ c_prev + β c_in."""
    dot = float(np.dot(c_prev, c_in))
    rho = _rho(beta, dot)
    return rho * c_prev + beta * c_in


def _log_softmax_masked(scaled, keep_mask):
    """Log-softmax over keep_mask=True; masked positions get -inf."""
    s = np.where(keep_mask, scaled, -1.0e30)
    m = s.max()
    shifted = s - m
    exp_s = np.where(keep_mask, np.exp(shifted), 0.0)
    denom = max(exp_s.sum(), 1e-300)
    return s - m - np.log(denom)


def _p_stop(m_cf_exp, c_ret, already_mask, eps_d):
    """Eq 7: p_stop = exp(-ε_d · a^nr / a^r) using |a| sums."""
    a = m_cf_exp @ c_ret
    a_abs = np.abs(a)
    a_r = float(np.sum(np.where(already_mask, a_abs, 0.0)))
    a_nr = float(np.sum(np.where(already_mask, 0.0, a_abs)))
    if a_r <= 0:
        return 0.0
    return float(np.exp(-eps_d * a_nr / a_r))


def _c_IN_rec(idx, m_fc_exp, gamma_fc, d):
    """Eq 4: c^IN_rec = (1-γ_fc) e_{idx+1} + γ_fc M^FC_exp[:, idx]."""
    pre = _onehot(d, idx + 1)
    return (1.0 - gamma_fc) * pre + gamma_fc * m_fc_exp[:, idx]


def _oracle_list_ll(
    params: ModelParameters,
    recall_sps: np.ndarray,   # 1-based; entries past R_valid may be padding
    recall_mask: np.ndarray,  # True for valid (non-padding) slots
    W: int,
) -> float:
    """Independent naive oracle for the C&Z marginalized list LL."""
    d = W + 1
    beta_enc = float(params.beta_enc)
    beta_list = float(params.beta_list)
    beta_rec = float(params.beta_rec)
    gamma_fc = float(params.gamma_fc)
    k = float(params.k)
    eps_d = float(params.epsilon_d)

    # --- ENCODING (Eqs 1, 2a, 2b, 8) ---
    c_item = _onehot(d, 0)  # e_start
    c_list = _onehot(d, 0)
    m_fc_exp = np.zeros((d, W), dtype=np.float64)
    m_cf_exp = np.zeros((W, d), dtype=np.float64)
    c_item_traj = [c_item.copy()]
    for t in range(W):
        # Eq 2a, 2b: pre-drift c_item.
        m_fc_exp[:, t] += c_item
        m_cf_exp[t, :] += c_item
        # c^IN_enc = e_{t+1} (identity M^FC_pre).
        c_in = _onehot(d, t + 1)
        c_item = _drift(c_item, c_in, beta_enc)   # Eq 1
        c_list = _drift(c_list, c_in, beta_list)  # Eq 8
        c_item_traj.append(c_item.copy())

    e_start = _onehot(d, 0)
    R_total = recall_sps.shape[0]

    # --- For each phase-transition T ∈ {0..R_total}, compute LL_T. ---
    # Track observed already_mask trajectory (independent of T). Repeats
    # (sp already in mask) are noise — skip mask update.
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

    # --- Phase-1 forward (cued by c_item_end). ---
    c_ret_phase1_traj = [c_item_traj[W].copy()]
    cum_phase1 = [0.0]
    cum = 0.0
    c_ret = c_item_traj[W].copy()
    for i in range(R_total):
        valid = bool(recall_mask[i])
        sp = int(recall_sps[i])
        idx = max(0, min(W - 1, sp - 1))
        already = mask_traj[i]  # mask BEFORE this slot's recall
        is_repeat = bool(already[idx])
        countable = valid and not is_repeat
        if countable:
            p_stop = _p_stop(m_cf_exp, c_ret, already, eps_d)
            log_1m = float(np.log(max(1.0 - p_stop, 1e-300)))
            a = m_cf_exp @ c_ret
            log_p = _log_softmax_masked(k * a, ~already)
            cum += log_1m + float(log_p[idx])
            c_in = _c_IN_rec(idx, m_fc_exp, gamma_fc, d)
            c_ret = _drift(c_ret, c_in, beta_rec)
        cum_phase1.append(cum)
        c_ret_phase1_traj.append(c_ret.copy())

    # --- For each T, run phase-2 starting at slot T from e_start. ---
    ll_per_T = np.zeros(R_total + 1, dtype=np.float64)
    for T in range(R_total + 1):
        # Phase-1 cumulative LL through slot T.
        phase1_ll = cum_phase1[T]
        # Transition log-prob at c_ret after T phase-1 drifts, with mask
        # = mask_traj[T].
        c_ret_at_trans = c_ret_phase1_traj[T]
        mask_at_trans = mask_traj[T]
        p_stop_trans = _p_stop(m_cf_exp, c_ret_at_trans, mask_at_trans, eps_d)
        log_trans = float(np.log(max(p_stop_trans, 1e-300)))

        # Phase-2 forward from slot T. Repeats are noise (no contribution).
        c_ret = e_start.copy()
        mask = mask_at_trans.copy()
        phase2_ll = 0.0
        for i in range(T, R_total):
            valid = bool(recall_mask[i])
            sp = int(recall_sps[i])
            idx = max(0, min(W - 1, sp - 1))
            is_repeat = bool(mask[idx])
            countable = valid and not is_repeat
            if countable:
                p_stop = _p_stop(m_cf_exp, c_ret, mask, eps_d)
                log_1m = float(np.log(max(1.0 - p_stop, 1e-300)))
                a = m_cf_exp @ c_ret
                log_p = _log_softmax_masked(k * a, ~mask)
                phase2_ll += log_1m + float(log_p[idx])
                c_in = _c_IN_rec(idx, m_fc_exp, gamma_fc, d)
                c_ret = _drift(c_ret, c_in, beta_rec)
                mask = mask.copy()
                mask[idx] = True

        # Final stop log-prob.
        p_stop_final = _p_stop(m_cf_exp, c_ret, mask, eps_d)
        log_final = float(np.log(max(p_stop_final, 1e-300)))

        ll_per_T[T] = phase1_ll + log_trans + phase2_ll + log_final

    # logsumexp.
    m = ll_per_T.max()
    return float(m + np.log(np.sum(np.exp(ll_per_T - m))))


# -----------------------------------------------------------------------
# Property tests: oracle vs core on random small lists.
# -----------------------------------------------------------------------


def _random_recalls(rng, W, max_R=None):
    """Generate a random valid recall sequence of length 1..max(2,W)."""
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


@pytest.mark.parametrize("W", [3, 4, 6, 10])
@pytest.mark.parametrize("seed", list(range(5)))
def test_core_matches_oracle_random(W, seed):
    """Random valid recall sequences: core LL must match naive oracle to 1e-10."""
    rng = np.random.default_rng(seed * 17 + W)
    recall_sps, recall_mask = _random_recalls(rng, W)
    params = ModelParameters(
        beta_enc=0.679, beta_story=0.400, gamma_fc=0.315, k=6.50,
        beta_rec=0.326, epsilon_d=1.04, beta_rein=0.300,
        lambda_reinstate=0.0, paradigm="free_recall",
    )
    ll_oracle = _oracle_list_ll(params, recall_sps, recall_mask, W=W)
    ll_core = compute_list_log_likelihood_numpy(
        params, recall_sps, recall_mask, W=W,
    )
    assert np.isclose(ll_oracle, ll_core, atol=1e-10, rtol=0), (
        f"core diverges from oracle: oracle={ll_oracle!r}, core={ll_core!r}, "
        f"diff={ll_oracle - ll_core!r}\n"
        f"  W={W}, seed={seed}, recall_sps={recall_sps.tolist()}, "
        f"mask={recall_mask.tolist()}"
    )


@pytest.mark.parametrize("seed", list(range(5)))
def test_core_matches_oracle_with_padding(seed):
    """Padded recall arrays (some valid, some padding) must match oracle."""
    rng = np.random.default_rng(seed)
    W = 8
    R_valid = 4
    R_pad = 12  # over-pad
    sps = (rng.permutation(W)[:R_valid] + 1).tolist()
    recall_sps = np.zeros(R_pad, dtype=np.int64)
    recall_mask = np.zeros(R_pad, dtype=bool)
    recall_sps[:R_valid] = sps
    recall_mask[:R_valid] = True
    params = ModelParameters(
        beta_enc=0.679, beta_story=0.400, gamma_fc=0.315, k=6.50,
        beta_rec=0.326, epsilon_d=1.04, beta_rein=0.300,
        lambda_reinstate=0.0, paradigm="free_recall",
    )
    ll_oracle = _oracle_list_ll(params, recall_sps, recall_mask, W=W)
    ll_core = compute_list_log_likelihood_numpy(
        params, recall_sps, recall_mask, W=W,
    )
    assert np.isclose(ll_oracle, ll_core, atol=1e-10, rtol=0), (
        f"oracle={ll_oracle!r}, core={ll_core!r}, diff={ll_oracle - ll_core!r}"
    )


def test_core_matches_oracle_single_recall():
    """One valid recall + 0 padding (no transition possibilities except T=0,1)."""
    params = ModelParameters(
        beta_enc=0.679, beta_story=0.400, gamma_fc=0.315, k=6.50,
        beta_rec=0.326, epsilon_d=1.04, beta_rein=0.300,
        lambda_reinstate=0.0, paradigm="free_recall",
    )
    W = 5
    recall_sps = np.array([3], dtype=np.int64)
    recall_mask = np.array([True], dtype=bool)
    ll_oracle = _oracle_list_ll(params, recall_sps, recall_mask, W=W)
    ll_core = compute_list_log_likelihood_numpy(
        params, recall_sps, recall_mask, W=W,
    )
    assert np.isclose(ll_oracle, ll_core, atol=1e-10, rtol=0)


def test_core_matches_oracle_all_W_recalled():
    """Recall every position; tests termination edge cases."""
    params = ModelParameters(
        beta_enc=0.679, beta_story=0.400, gamma_fc=0.315, k=6.50,
        beta_rec=0.326, epsilon_d=1.04, beta_rein=0.300,
        lambda_reinstate=0.0, paradigm="free_recall",
    )
    W = 5
    recall_sps = np.array([5, 4, 3, 2, 1], dtype=np.int64)
    recall_mask = np.array([True, True, True, True, True])
    ll_oracle = _oracle_list_ll(params, recall_sps, recall_mask, W=W)
    ll_core = compute_list_log_likelihood_numpy(
        params, recall_sps, recall_mask, W=W,
    )
    assert np.isclose(ll_oracle, ll_core, atol=1e-10, rtol=0)


@pytest.mark.parametrize("beta_enc,beta_list,k,beta_rec,eps_d,gamma_fc", [
    (0.3, 0.2, 2.0, 0.2, 0.5, 0.1),
    (0.9, 0.5, 15.0, 0.7, 3.0, 0.9),
    (0.5, 0.2, 6.5, 0.3, 1.5, 0.5),
])
def test_core_matches_oracle_param_sweep(
    beta_enc, beta_list, k, beta_rec, eps_d, gamma_fc,
):
    """Vary parameters across plausible ranges; oracle and core agree."""
    params = ModelParameters(
        beta_enc=beta_enc, beta_story=beta_list, gamma_fc=gamma_fc, k=k,
        beta_rec=beta_rec, epsilon_d=eps_d, beta_rein=0.300,
        lambda_reinstate=0.0, paradigm="free_recall",
    )
    rng = np.random.default_rng(7)
    W = 6
    recall_sps, recall_mask = _random_recalls(rng, W, max_R=4)
    ll_oracle = _oracle_list_ll(params, recall_sps, recall_mask, W=W)
    ll_core = compute_list_log_likelihood_numpy(
        params, recall_sps, recall_mask, W=W,
    )
    assert np.isclose(ll_oracle, ll_core, atol=1e-10, rtol=0), (
        f"params={params!r}, recall_sps={recall_sps.tolist()}: "
        f"oracle={ll_oracle!r}, core={ll_core!r}, diff={ll_oracle - ll_core!r}"
    )
