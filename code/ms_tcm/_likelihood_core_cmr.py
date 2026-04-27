"""Polyn et al. 2009 standard (non-hierarchical) CMR for free recall.

Reference: Polyn, Norman, & Kahana (2009). A context maintenance and
retrieval model of organizational processes in free recall. Psych Rev,
116(1), 129-156.

This is the "standard" CMR baseline used in the MS-TCM model comparison
(alongside C&Z 2025 hierarchical CMR). It differs from C&Z 2025 in three
ways:

1. Single context (no list-level c_list): the model uses one c_item that
   drifts at rate β_enc throughout encoding, and one c_ret that drifts
   at β_rec during retrieval.
2. Explicit primacy gradient at retrieval:
   ``a_j = (φ_l ⊙ M^CF_exp) · c_ret`` with column-l weights
   ``φ_l = φ_s · exp(-φ_d · (l - 1)) + 1`` (Polyn et al. 2009 Eq 5).
3. Single retrieval phase (no T-transition marginalization): retrieval
   begins from c_ret = c_item_end and continues until the stopping rule
   fires.

Free parameters (8 for the curve-fit comparison):
    β_enc, β_rec, γ_fc, k, ε_d, φ_s, φ_d  (plus the non-free β_rein=0)

NOT exported as a likelihood — only as a simulator. The MS-TCM curve-
matching fitter doesn't need a closed-form LL for any of the three
models in the comparison; all three are evaluated at the curve level.
"""

from __future__ import annotations

from dataclasses import dataclass

from ms_tcm._likelihood_shared import (
    _is_jax,
    activation, c_IN_rec_of, compute_p_stop, drift_c_ret, log_softmax_masked,
    norm_preserving_rho, at_add, at_set,
)


@dataclass(frozen=True)
class CMRCoreHyperparams:
    """Polyn 2009 standard-CMR free-recall parameters."""

    beta_enc: float
    beta_rec: float
    gamma_fc: float
    k: float
    epsilon_d: float
    phi_s: float
    phi_d: float

    @classmethod
    def from_model_parameters(cls, p) -> "CMRCoreHyperparams":
        return cls(
            beta_enc=float(p.beta_enc),
            beta_rec=float(p.beta_rec),
            gamma_fc=float(p.gamma_fc),
            k=float(p.k),
            epsilon_d=float(p.epsilon_d),
            phi_s=float(p.phi_s),
            phi_d=float(p.phi_d),
        )


def run_encoding_cmr(hp: CMRCoreHyperparams, W: int, xp, dtype):
    """Encode under Polyn 2009 (single c_item drift; no list-level).

    Returns (c_item_traj, M_fc_exp, M_cf_exp) — same signature as
    `run_encoding` in _likelihood_core.py minus the c_list bookkeeping.
    Supports both numpy and jax.numpy via the `xp` namespace.
    """
    d = W + 1
    e_start = xp.zeros(d, dtype=dtype)
    e_start = at_set(e_start, 0, 1.0, xp)
    c_item = e_start
    m_fc_exp = xp.zeros((d, W), dtype=dtype)
    m_cf_exp = xp.zeros((W, d), dtype=dtype)

    if _is_jax(xp):
        import jax

        def step(carry, t):
            c_item, m_fc_exp, m_cf_exp = carry
            m_fc_exp = at_set(m_fc_exp, (slice(None), t), m_fc_exp[:, t] + c_item, xp)
            m_cf_exp = at_set(m_cf_exp, (t, slice(None)), m_cf_exp[t, :] + c_item, xp)
            t_plus_1 = t + 1
            dot_i = c_item[t_plus_1]
            rho_i = norm_preserving_rho(hp.beta_enc, dot_i, xp)
            c_item = rho_i * c_item
            c_item = at_add(c_item, t_plus_1, hp.beta_enc, xp)
            return (c_item, m_fc_exp, m_cf_exp), c_item

        t_arr = xp.arange(W, dtype=xp.int32)
        init = (c_item, m_fc_exp, m_cf_exp)
        (_, m_fc_final, m_cf_final), c_item_steps = jax.lax.scan(step, init, t_arr)
        c_item_traj = xp.concatenate([e_start[None, :], c_item_steps], axis=0)
        return c_item_traj, m_fc_final, m_cf_final

    # numpy path.
    c_item_traj = xp.zeros((W + 1, d), dtype=dtype)
    c_item_traj = at_set(c_item_traj, 0, e_start, xp)
    for t in range(W):
        m_fc_exp = at_set(m_fc_exp, (slice(None), t), m_fc_exp[:, t] + c_item, xp)
        m_cf_exp = at_set(m_cf_exp, (t, slice(None)), m_cf_exp[t, :] + c_item, xp)
        t_plus_1 = t + 1
        dot_i = c_item[t_plus_1]
        rho_i = norm_preserving_rho(hp.beta_enc, dot_i, xp)
        c_item = rho_i * c_item
        c_item = at_add(c_item, t_plus_1, hp.beta_enc, xp)
        c_item_traj = at_set(c_item_traj, t + 1, c_item, xp)
    return c_item_traj, m_fc_exp, m_cf_exp


def compute_list_log_likelihood_cmr(
    hp: CMRCoreHyperparams,
    W: int,
    recall_sps,    # (R,) int: 1-based serial positions; 0 = padding
    recall_mask,   # (R,) bool: True for valid recall slots
    xp,
    dtype,
):
    """Per-list log-likelihood under Polyn 2009 standard CMR.

    Single-phase retrieval starting from c_ret = c_item_end. For each
    valid recall slot i:

        contrib_i = log(1 - p_stop_i)            # continued
                   + log_softmax(k · a_i)[s_i - 1]
        c_ret    ← drift(c_ret, c_IN_rec(s_i - 1), β_rec)

    After all R drifts, add log p_stop_final at the end.

    No T-marginalization (CMR has only one phase). Works in numpy or
    JAX via the `xp` namespace; gradient-friendly under jax.grad.
    """
    d = W + 1
    c_item_traj, m_fc_exp, m_cf_exp = run_encoding_cmr(hp, W, xp, dtype)
    phi = primacy_gradient(W, hp.phi_s, hp.phi_d, xp, dtype)  # (W,)
    m_cf_eff = m_cf_exp * phi[:, None]  # (W, d) — broadcast over context dim

    c_ret_init = c_item_traj[W]
    R_pad = recall_sps.shape[0]

    def step(carry, t):
        c_ret, cum, already = carry
        sp = recall_sps[t]
        valid = recall_mask[t]
        # Index for sp-1 (clamped to valid range so JAX gather doesn't
        # complain on padded slots).
        sp_idx = xp.clip(sp - 1, 0, W - 1)
        is_repeat = already[sp_idx]
        countable = valid & ~is_repeat
        # Activations + p_stop computed at PRE-drift c_ret.
        a = activation(m_cf_eff, c_ret, xp)
        p_stop = compute_p_stop(m_cf_eff, c_ret, already, hp.epsilon_d, xp)
        log_continue = xp.log(xp.maximum(1.0 - p_stop, 1e-300))
        log_p_pick = log_softmax_masked(hp.k * a, ~already, xp)
        contrib = xp.where(
            countable,
            log_continue + log_p_pick[sp_idx],
            xp.asarray(0.0, dtype=dtype),
        )
        cum = cum + contrib
        # Drift only if countable.
        c_in = c_IN_rec_of(sp_idx, m_fc_exp, hp.gamma_fc, d, xp, dtype)
        c_ret_new = drift_c_ret(c_ret, c_in, hp.beta_rec, xp)
        c_ret = xp.where(countable, c_ret_new, c_ret)
        # already_mask updated only when countable.
        already_new = at_set(already, sp_idx, True, xp) if not _is_jax(xp) else already.at[sp_idx].set(True)
        already = xp.where(countable, already_new, already)
        return (c_ret, cum, already), None

    cum0 = xp.asarray(0.0, dtype=dtype)
    already0 = xp.zeros(W, dtype=bool)
    init = (c_ret_init, cum0, already0)
    if _is_jax(xp):
        import jax
        (c_ret_final, cum_total, already_final), _ = jax.lax.scan(
            step, init, xp.arange(R_pad, dtype=xp.int32),
        )
    else:
        carry = init
        for t in range(R_pad):
            carry, _ = step(carry, t)
        c_ret_final, cum_total, already_final = carry

    # Final stopping term.
    p_stop_final = compute_p_stop(
        m_cf_eff, c_ret_final, already_final, hp.epsilon_d, xp,
    )
    log_p_stop_final = xp.log(xp.maximum(p_stop_final, 1e-300))
    return cum_total + log_p_stop_final


def primacy_gradient(W: int, phi_s: float, phi_d: float, xp, dtype):
    """φ_l = φ_s · exp(-φ_d · (l-1)) + 1 for l = 1..W (Polyn 2009 Eq 5)."""
    l_idx = xp.arange(W, dtype=dtype)  # 0-indexed; l-1 in the formula
    return phi_s * xp.exp(-phi_d * l_idx) + 1.0


def simulate_recalls_cmr(params, W: int, rng, *, max_recalls: int | None = None):
    """Sample a free-recall sequence from Polyn 2009 standard CMR.

    Single-phase retrieval starting from c_ret = c_item_end. Activations
    are scaled column-wise by the primacy gradient φ_l (Polyn Eq 5).
    """
    import numpy as np

    hp = CMRCoreHyperparams.from_model_parameters(params)
    d = W + 1
    c_item_traj, m_fc_exp, m_cf_exp = run_encoding_cmr(hp, W, np, np.float64)
    phi = primacy_gradient(W, hp.phi_s, hp.phi_d, np, np.float64)  # (W,)
    # Polyn Eq 5: φ multiplies M^CF_exp on the row axis (one weight per item).
    # Equivalently, multiply each row of M^CF_exp by φ[l].
    m_cf_eff = m_cf_exp * phi[:, None]

    cap = max_recalls if max_recalls is not None else 3 * W
    recalls: list[int] = []
    already_mask = np.zeros(W, dtype=bool)
    c_ret = c_item_traj[W].copy()  # end-of-list cue
    while len(recalls) < cap:
        p_stop = float(compute_p_stop(
            m_cf_eff, c_ret, already_mask, hp.epsilon_d, np,
        ))
        if rng.random() < p_stop:
            break
        a = activation(m_cf_eff, c_ret, np)
        log_p = log_softmax_masked(hp.k * a, ~already_mask, np)
        p = np.exp(log_p - log_p.max())
        p = np.where(already_mask, 0.0, p)
        if p.sum() <= 0:
            break
        p = p / p.sum()
        idx = int(rng.choice(W, p=p))
        recalls.append(idx + 1)
        already_mask[idx] = True
        c_in = c_IN_rec_of(idx, m_fc_exp, hp.gamma_fc, d, np, np.float64)
        c_ret = drift_c_ret(c_ret, c_in, hp.beta_rec, np)
    return recalls
