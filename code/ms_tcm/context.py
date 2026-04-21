"""Global and storyline context updates (§3.2).

Pure functions; all list-boundary / list-bookkeeping is the caller's
responsibility (the orchestrator in ``ms_tcm.model``).
"""

from __future__ import annotations

import numpy as np


def update_global_context(
    c_prev: np.ndarray, beta_global: float, c_in: np.ndarray,
) -> np.ndarray:
    """c_G(t) = rho_G * c_G(t-1) + beta_G * c^IN_i, with rho_G = sqrt(1 - beta_G^2)."""
    if not (0.0 < beta_global < 1.0):
        raise ValueError(f"beta_global must be in (0, 1); got {beta_global!r}")
    c_prev = np.asarray(c_prev, dtype=np.float64)
    c_in = np.asarray(c_in, dtype=np.float64)
    if c_prev.shape != c_in.shape:
        raise ValueError(
            f"update_global_context: shape mismatch {c_prev.shape} vs {c_in.shape}"
        )
    rho = np.sqrt(1.0 - beta_global * beta_global)
    return rho * c_prev + beta_global * c_in


def update_storyline_context(
    c_prev: np.ndarray, beta_storyline: float, c_in: np.ndarray,
) -> np.ndarray:
    """c_S(t) = rho_S * c_S(t_prev_S) + beta_S * c^IN_i, with rho_S = sqrt(1 - beta_S^2).

    Callers MUST only invoke this when storyline S is active at the current
    step (i.e., the encoded event belongs to storyline S). Inactive storylines
    are frozen: the caller propagates the previous value unchanged.
    """
    if not (0.0 < beta_storyline < 1.0):
        raise ValueError(f"beta_storyline must be in (0, 1); got {beta_storyline!r}")
    c_prev = np.asarray(c_prev, dtype=np.float64)
    c_in = np.asarray(c_in, dtype=np.float64)
    if c_prev.shape != c_in.shape:
        raise ValueError(
            f"update_storyline_context: shape mismatch {c_prev.shape} vs {c_in.shape}"
        )
    rho = np.sqrt(1.0 - beta_storyline * beta_storyline)
    return rho * c_prev + beta_storyline * c_in
