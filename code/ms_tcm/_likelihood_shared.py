"""Shared math primitives used by C&Z and MS-TCM likelihood cores.

Both ``_likelihood_core.py`` (C&Z 2025 hierarchical free-recall) and
``_likelihood_core_mstcm.py`` (multi-storyline TCM extension) import from
this module. Keeping these helpers in one place enforces Constitution II
(Single Source of Truth): a numerical fix in any of these primitives
applies identically to both models.

All functions take an ``xp`` namespace (numpy or jax.numpy) so the same
code paths drive Tier-1 and Tier-2 backends.
"""

from __future__ import annotations

from typing import Any


# --- xp dispatch helpers -------------------------------------------------


def _is_jax(xp: Any) -> bool:
    return getattr(xp, "__name__", "") == "jax.numpy"


def at_set(x, idx, value, xp):
    """Functional scatter: return a new array with ``x[idx] = value``."""
    if _is_jax(xp):
        return x.at[idx].set(value)
    x = x.copy()
    x[idx] = value
    return x


def at_add(x, idx, delta, xp):
    """Functional scatter-add: return a new array with ``x[idx] += delta``."""
    if _is_jax(xp):
        return x.at[idx].add(delta)
    x = x.copy()
    x[idx] = x[idx] + delta
    return x


# --- numerical primitives ------------------------------------------------


def log_softmax_masked(scaled, keep_mask, xp):
    """Numerically stable log-softmax over keep_mask=True entries.

    ``mask=False`` positions get a large negative sentinel and contribute 0
    to the denominator. Result for masked positions is unspecified; callers
    must only index keep-mask positions.
    """
    NEG = -1.0e30
    s = xp.where(keep_mask, scaled, NEG)
    m = xp.max(s)
    shifted = s - m
    exp_s = xp.where(keep_mask, xp.exp(shifted), 0.0)
    denom = xp.sum(exp_s)
    return s - m - xp.log(xp.maximum(denom, 1e-300))


def norm_preserving_rho(beta, dot, xp):
    """ρ such that ||ρ c_prev + β c_in|| = 1 for unit-norm c_prev, c_in."""
    inner = 1.0 + beta * beta * (dot * dot - 1.0)
    return xp.sqrt(xp.maximum(inner, 0.0)) - beta * dot


# --- retrieval helpers ---------------------------------------------------


def activation(m_cf_exp, c_ret, xp):
    """C&Z Eq 5: a = M^CF_exp · c_ret. Shape: (W,)."""
    return m_cf_exp @ c_ret


def compute_p_stop(m_cf_exp, c_ret, already_mask, epsilon_d, xp):
    """C&Z Eq 7: p_stop = exp(-ε_d · a^nr / a^r) using |a| sums."""
    a = activation(m_cf_exp, c_ret, xp)
    a_abs = xp.abs(a)
    a_r = xp.sum(xp.where(already_mask, a_abs, 0.0))
    a_nr = xp.sum(xp.where(already_mask, 0.0, a_abs))
    ratio = a_nr / xp.maximum(a_r, 1e-30)
    p_stop_raw = xp.exp(-epsilon_d * ratio)
    return xp.where(a_r > 0.0, p_stop_raw, 0.0)


def drift_c_ret(c_ret, c_IN_rec, beta_rec, xp):
    """C&Z Eq 3: c^ret = ρ · c^ret + β_rec · c^IN_rec."""
    dot = xp.sum(c_ret * c_IN_rec)
    rho = norm_preserving_rho(beta_rec, dot, xp)
    return rho * c_ret + beta_rec * c_IN_rec


def c_IN_rec_of(idx, m_fc_exp, gamma_fc, d, xp, dtype):
    """C&Z Eq 4: c^IN_rec = (1-γ_fc) M^FC_pre · f_j + γ_fc M^FC_exp · f_j.

    Identity M^FC_pre: M^FC_pre · f_j = e_{idx+1} (basis vector). The
    experimental branch reads column M^FC_exp[:, idx].
    """
    pre = xp.zeros(d, dtype=dtype)
    pre = at_set(pre, idx + 1, 1.0, xp)
    exp_col = m_fc_exp[:, idx]
    return (1.0 - gamma_fc) * pre + gamma_fc * exp_col
