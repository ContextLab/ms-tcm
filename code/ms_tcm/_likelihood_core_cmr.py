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
    """
    d = W + 1
    e_start = xp.zeros(d, dtype=dtype)
    e_start = at_set(e_start, 0, 1.0, xp)
    c_item = e_start
    m_fc_exp = xp.zeros((d, W), dtype=dtype)
    m_cf_exp = xp.zeros((W, d), dtype=dtype)
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
