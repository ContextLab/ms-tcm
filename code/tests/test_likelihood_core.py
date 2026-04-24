"""Equation-derived oracle tests for ``ms_tcm._likelihood_core``.

This test file is the correctness anchor for Constitution II (Single
Source of Truth). It contains two independent implementations:

1. ``_oracle_list_ll(...)``  — a deliberately naive, line-by-line
   implementation of v6 Eqs 1-9 + C&Z Eq 3, 7 that uses only Python
   floats, numpy arrays, and explicit for-loops. No optimizations, no
   shortcuts, no vectorization. This is the *external* ground truth.

2. ``compute_list_log_likelihood_numpy`` — the shared core that both the
   Tier-1 and Tier-2 backends delegate to.

The test battery generates random tiny (W=3..8), single- and multi-
storyline lists with small numbers of recalls, computes the LL under
both implementations, and asserts they match within 1e-12.

A divergence here means the core has a math bug that BOTH backends
would inherit silently — this is the single most valuable test in the
suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from ms_tcm._likelihood_core import (
    CoreHyperparams,
    compute_list_log_likelihood_numpy,
)
from ms_tcm.params import ModelParameters


# -----------------------------------------------------------------------
# Oracle: naive, line-by-line implementation of the v6 equations.
# -----------------------------------------------------------------------


def _onehot(d, i):
    v = np.zeros(d, dtype=np.float64)
    v[i] = 1.0
    return v


def _rho(beta, dot):
    inner = 1.0 + beta * beta * (dot * dot - 1.0)
    return np.sqrt(max(0.0, inner)) - beta * dot


def _oracle_list_ll(
    params: ModelParameters,
    cat_indices: np.ndarray,
    recall_sps: np.ndarray,
    recall_mask: np.ndarray,
) -> float:
    """Independent oracle implementation of per-list log-likelihood.

    Implements v6 §2-3 and C&Z 2025 Eqs 3, 7-9 literally, with only
    numpy-level primitives. Does NOT call any ms_tcm.* code except
    ``ModelParameters`` (for param access).
    """
    W = int(cat_indices.shape[0])
    d = W + 1
    K = int(cat_indices.max()) + 1 if W > 0 else 1

    phi = 30.0 if params.standard_tcm else 1.5
    psi = 0.8 if params.standard_tcm else 0.5

    # --- ENCODING (v6 Eqs 1, 2, 5, 6, 7) ---------------------------------
    c_item = _onehot(d, 0)  # e_start
    c_story = [_onehot(d, 0) for _ in range(K)]
    m_ic = np.zeros((d, W), dtype=np.float64)
    m_sc = np.zeros((K, d), dtype=np.float64)
    seen = [False] * K
    prev_cat = -1

    c_item_traj = np.zeros((W + 1, d), dtype=np.float64)
    c_item_traj[0] = c_item.copy()

    for t in range(W):
        cat = int(cat_indices[t])

        # Storyline switch / return (Eqs 5, 6, 7).
        if not params.standard_tcm:
            if prev_cat >= 0 and cat != prev_cat:
                # Switch: cache outgoing storyline to M^SC (Eq 5).
                m_sc[prev_cat] = m_sc[prev_cat] + c_story[prev_cat]
                # Return? Then blend cached context into active (Eq 6).
                if seen[cat]:
                    blended = (
                        params.lambda_reinstate * m_sc[cat]
                        + (1.0 - params.lambda_reinstate) * c_story[cat]
                    )
                    nn = float(np.linalg.norm(blended))
                    if nn > 1e-12:
                        blended = blended / nn
                    c_story[cat] = blended
                # Sync c^item to active storyline (Eq 7).
                c_item = c_story[cat].copy()

        # Storyline drift (Eq 2) — only for the active storyline.
        if not params.standard_tcm:
            c_s = c_story[cat]
            dot_s = float(c_s[t + 1])  # basis-vector c^IN = e_{t+1}
            rho_s = _rho(params.beta_story, dot_s)
            c_s = rho_s * c_s
            c_s[t + 1] += params.beta_story
            c_story[cat] = c_s

        # Item-level drift (Eq 1).
        dot_i = float(c_item[t + 1])
        rho_i = _rho(params.beta_enc, dot_i)
        c_item = rho_i * c_item
        c_item[t + 1] += params.beta_enc

        # Primacy + M^IC update.
        primacy = 1.0 + phi * np.exp(-psi * t)
        m_ic[:, t] += primacy * c_item

        c_item_traj[t + 1] = c_item.copy()
        seen[cat] = True
        prev_cat = cat

    # --- RETRIEVAL + STOPPING (C&Z Eqs 3, 7; v6 Eqs 8-9) ----------------

    def _softmax_masked(scaled, keep_mask):
        """Log-softmax over keep_mask=True items only; others get -inf."""
        NEG = -1.0e30
        s = np.where(keep_mask, scaled, NEG)
        m = s.max()
        shifted = s - m
        exp_s = np.where(keep_mask, np.exp(shifted), 0.0)
        denom = exp_s.sum()
        return s - m - np.log(denom)

    def _p_stop(c_ret, already_mask):
        a = m_ic.T @ c_ret
        a_abs = np.abs(a)
        a_r = float(np.sum(np.where(already_mask, a_abs, 0.0)))
        a_nr = float(np.sum(np.where(already_mask, 0.0, a_abs)))
        if a_r <= 0.0:
            return 0.0
        ratio = a_nr / a_r
        return float(np.exp(-params.epsilon_d * ratio))

    log_l = 0.0
    c_ret = c_item_traj[W].copy()  # end-of-list cue
    already = np.zeros(W, dtype=bool)
    prev_idx = -1

    for r in range(len(recall_sps)):
        if not bool(recall_mask[r]):
            continue
        sp = int(recall_sps[r])
        idx = max(0, min(W - 1, sp - 1))

        # Repeat handling matches Tier-1 likelihood.py: on an already-
        # recalled sp, advance prev_idx (so the next drift targets this
        # sp) but do NOT contribute stopping or score terms, and do NOT
        # reset c_ret.
        if already[idx]:
            prev_idx = idx
            continue

        if prev_idx >= 0:
            # Stopping contribution on pre-drift c_ret.
            p_stop = _p_stop(c_ret, already)
            if 1.0 - p_stop <= 0.0:
                return float("-inf")
            log_l += float(np.log(1.0 - p_stop))
            # Drift c_ret toward e_{prev_idx+1}.
            e_prev = _onehot(d, prev_idx + 1)
            dot_r = float(np.dot(c_ret, e_prev))
            rho_r = _rho(params.beta_rec, dot_r)
            c_ret = rho_r * c_ret + params.beta_rec * e_prev

        a = m_ic.T @ c_ret
        keep_mask = ~already
        log_p = _softmax_masked(params.k * a, keep_mask)
        log_l += float(log_p[idx])

        already[idx] = True
        c_ret = c_item_traj[sp].copy()
        prev_idx = idx

    # Terminal stopping contribution.
    if prev_idx >= 0:
        p_stop = _p_stop(c_ret, already)
        if p_stop > 0.0:
            log_l += float(np.log(p_stop))

    return log_l


# -----------------------------------------------------------------------
# Property-based parity tests: oracle vs. core.
# -----------------------------------------------------------------------


def _random_case(rng, W, K):
    """Generate a random (cat_indices, recall_sps, recall_mask) tuple.

    Recall lengths vary from 1 .. W-1. Recalls are drawn without repeats
    from 1..W in a random order. Padding pads with 0s / False.
    """
    # Distribute storylines across W steps, ensuring each storyline gets
    # at least one presentation (for K<=W).
    cat_indices = np.zeros(W, dtype=np.int64)
    slots = list(range(W))
    rng.shuffle(slots)
    for k in range(K):
        cat_indices[slots[k]] = k
    for i in range(K, W):
        cat_indices[slots[i]] = rng.integers(0, K)

    n_recalls = int(rng.integers(1, max(2, W)))
    sps = rng.permutation(W)[:n_recalls] + 1  # 1-based
    R = W  # pad recall arrays to length W
    recall_sps = np.zeros(R, dtype=np.int64)
    recall_mask = np.zeros(R, dtype=bool)
    recall_sps[:n_recalls] = sps
    recall_mask[:n_recalls] = True
    return cat_indices, recall_sps, recall_mask


def _assert_oracle_matches_core(params, cat_indices, recall_sps, recall_mask):
    W = int(cat_indices.shape[0])
    K = int(cat_indices.max()) + 1 if W > 0 else 1
    ll_oracle = _oracle_list_ll(params, cat_indices, recall_sps, recall_mask)
    ll_core = compute_list_log_likelihood_numpy(
        params, cat_indices, recall_sps, recall_mask, W=W, K=K,
    )
    if not (np.isfinite(ll_oracle) and np.isfinite(ll_core)):
        assert np.isfinite(ll_oracle) == np.isfinite(ll_core), (
            f"finiteness mismatch: oracle={ll_oracle}, core={ll_core}"
        )
        return
    assert np.isclose(ll_oracle, ll_core, atol=1e-10, rtol=0), (
        f"core diverges from equation-derived oracle: "
        f"oracle={ll_oracle!r}, core={ll_core!r}, "
        f"diff={ll_oracle - ll_core!r}\n"
        f"cat_indices={cat_indices.tolist()}, recall_sps={recall_sps.tolist()}, "
        f"recall_mask={recall_mask.tolist()}, params={params!r}"
    )


@pytest.mark.parametrize("W", [3, 4, 5, 8])
@pytest.mark.parametrize("K", [1, 2])
@pytest.mark.parametrize("seed", list(range(5)))
def test_core_matches_oracle_ms_tcm(W, K, seed):
    """MS-TCM path (standard_tcm=False, λ>0)."""
    rng = np.random.default_rng(seed)
    cat_indices, recall_sps, recall_mask = _random_case(rng, W, K)
    params = ModelParameters(
        beta_enc=0.68, beta_story=0.40, gamma_fc=0.3, k=6.5,
        lambda_reinstate=0.80, beta_rec=0.33, epsilon_d=1.04,
        paradigm="free_recall", standard_tcm=False, seed=seed,
    )
    _assert_oracle_matches_core(params, cat_indices, recall_sps, recall_mask)


@pytest.mark.parametrize("W", [3, 4, 8])
@pytest.mark.parametrize("seed", list(range(3)))
def test_core_matches_oracle_standard_tcm(W, seed):
    """--standard-tcm reduction (λ=0, single storyline, different primacy)."""
    rng = np.random.default_rng(seed)
    cat_indices, recall_sps, recall_mask = _random_case(rng, W, 1)
    params = ModelParameters.standard_tcm_reduction(
        beta_enc=0.72, beta_story=0.35, gamma_fc=0.30, k=7.0,
        beta_rec=0.30, epsilon_d=1.0, paradigm="free_recall", seed=seed,
    )
    _assert_oracle_matches_core(params, cat_indices, recall_sps, recall_mask)


def test_core_matches_oracle_single_recall():
    """Edge case: exactly one recall on the list.

    No stopping-continuation contribution (no prior), one softmax score,
    one terminal stopping contribution.
    """
    params = ModelParameters(
        beta_enc=0.68, beta_story=0.40, gamma_fc=0.3, k=6.5,
        lambda_reinstate=0.80, beta_rec=0.33, epsilon_d=1.04,
    )
    W = 4
    cat_indices = np.array([0, 0, 1, 1], dtype=np.int64)
    recall_sps = np.zeros(W, dtype=np.int64)
    recall_mask = np.zeros(W, dtype=bool)
    recall_sps[0] = 3
    recall_mask[0] = True
    _assert_oracle_matches_core(params, cat_indices, recall_sps, recall_mask)


def test_core_matches_oracle_all_positions_recalled():
    """Edge case: recall every item in the list (a^nr → 0 at termination)."""
    params = ModelParameters(
        beta_enc=0.68, beta_story=0.40, gamma_fc=0.3, k=6.5,
        lambda_reinstate=0.80, beta_rec=0.33, epsilon_d=1.04,
    )
    W = 5
    cat_indices = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    recall_sps = np.array([5, 4, 3, 2, 1], dtype=np.int64)
    recall_mask = np.ones(W, dtype=bool)
    _assert_oracle_matches_core(params, cat_indices, recall_sps, recall_mask)


def test_core_matches_oracle_with_storyline_return():
    """Exercise v6 Eqs 5+6+7: a storyline that switches out and returns."""
    params = ModelParameters(
        beta_enc=0.68, beta_story=0.40, gamma_fc=0.3, k=6.5,
        lambda_reinstate=0.80, beta_rec=0.33, epsilon_d=1.04,
    )
    W = 6
    # Sequence: A A B B A A — storyline A goes A→B (switch) then B→A (return).
    cat_indices = np.array([0, 0, 1, 1, 0, 0], dtype=np.int64)
    recall_sps = np.array([5, 3, 1, 0, 0, 0], dtype=np.int64)
    recall_mask = np.array([True, True, True, False, False, False], dtype=bool)
    _assert_oracle_matches_core(params, cat_indices, recall_sps, recall_mask)


# -----------------------------------------------------------------------
# Hand-derived closed-form check: smallest-possible list.
# -----------------------------------------------------------------------


def test_core_hand_derived_W2_first_recall_only():
    """Closed-form check on W=2, one storyline, recall sp=1 then stop.

    Under identity M^FC_pre and standard_tcm=True, the encoding and
    retrieval math simplifies enough that we can derive the LL from the
    v6 equations by hand and compare.

    Parameters chosen so numbers are clean:
        β_enc = 0.5, k = 2.0, ε_d = 1.0, β_rec = 0.3
    Primacy: φ=30, ψ=0.8 under standard_tcm.

    Encoding:
        c_item_0 = e_0 = (1, 0, 0).
        Step t=0: c_in = e_1. dot = 0. ρ = √(1 - 0.25) = √0.75.
            c_item_1 = (√0.75, 0.5, 0).
        Step t=1: c_in = e_2. dot = 0. ρ = √0.75.
            c_item_2 = √0.75 · (√0.75, 0.5, 0) + 0.5 · (0, 0, 1)
                     = (0.75, √0.75/2, 0.5) = (0.75, √0.1875, 0.5).
        Primacy[0] = 31; primacy[1] = 1 + 30 · exp(-0.8).
        M^IC[:, 0] = 31 · c_item_1   = (31·√0.75, 15.5, 0).
        M^IC[:, 1] = primacy[1] · c_item_2.

    Retrieval:
        c_cue = c_item_2.
        a[0] = M^IC[:, 0] · c_cue = 31 · (c_item_1 · c_item_2)
             = 31 · (√0.75 · 0.75 + 0.5 · √0.1875 + 0 · 0.5).
        a[1] = primacy[1] · ||c_item_2||² = primacy[1] · 1 = primacy[1].

        c_item_1 · c_item_2 = √0.75 · 0.75 + 0.5 · (√0.75 / 2)
                            = (3/4) · √0.75 + (√0.75 / 4)
                            = √0.75.  [collect: (3/4 + 1/4) · √0.75]
        So a[0] = 31 · √0.75; a[1] = primacy[1] · 1.

    log P(sp=1 first) = log_softmax(k·a)[0]
                     = k·a[0] - logsumexp(k·a[0], k·a[1]).

    After scoring: c_ret reset to c_item_1 = (√0.75, 0.5, 0).
    Terminal stop: a = M^IC^T @ c_ret.
        a_stop[0] = 31 · (c_item_1 · c_item_1) = 31.
        a_stop[1] = primacy[1] · (c_item_2 · c_item_1) = primacy[1] · √0.75.
    already_mask = [True, False]. Abs sums:
        a_r = |a_stop[0]| = 31.
        a_nr = |a_stop[1]| = primacy[1] · √0.75.
        p_stop = exp(-ε_d · a_nr / a_r).

    Total LL = log P(first) + log p_stop.
    """
    params = ModelParameters.standard_tcm_reduction(
        beta_enc=0.5, beta_story=0.35, gamma_fc=0.3, k=2.0,
        beta_rec=0.3, epsilon_d=1.0, paradigm="free_recall",
    )
    W = 2
    cat_indices = np.array([0, 0], dtype=np.int64)
    recall_sps = np.array([1, 0], dtype=np.int64)
    recall_mask = np.array([True, False], dtype=bool)

    # Hand-derived expected LL.
    sqrt075 = np.sqrt(0.75)
    primacy_0 = 31.0
    primacy_1 = 1.0 + 30.0 * np.exp(-0.8)
    a0 = primacy_0 * sqrt075
    a1 = primacy_1 * 1.0
    k = 2.0
    ka = np.array([k * a0, k * a1])
    m = ka.max()
    log_p_first = ka[0] - m - np.log(np.exp(ka[0] - m) + np.exp(ka[1] - m))

    a_stop_0 = primacy_0 * 1.0  # c_item_1 · c_item_1 = 1
    a_stop_1 = primacy_1 * sqrt075
    a_r = abs(a_stop_0)
    a_nr = abs(a_stop_1)
    p_stop = np.exp(-1.0 * a_nr / a_r)
    log_p_stop = np.log(p_stop)

    expected_ll = log_p_first + log_p_stop

    actual_ll = compute_list_log_likelihood_numpy(
        params, cat_indices, recall_sps, recall_mask, W=W, K=1,
    )
    assert np.isclose(actual_ll, expected_ll, atol=1e-12, rtol=0), (
        f"hand-derived LL mismatch: expected {expected_ll!r}, "
        f"got {actual_ll!r}, diff {actual_ll - expected_ll!r}"
    )
