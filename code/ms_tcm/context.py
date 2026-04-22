"""Global and storyline context updates (§3.2).

Pure functions; all list-boundary / list-bookkeeping is the caller's
responsibility (the orchestrator in ``ms_tcm.model``).
"""

from __future__ import annotations

import numpy as np


def _norm_preserving_rho(beta: float, dot_prev_in: float) -> float:
    """Choose rho so that ``||rho * c_prev + beta * c_in|| = 1`` when both
    ``c_prev`` and ``c_in`` are unit-norm.

    Expanding ``||rho * c_prev + beta * c_in||^2 = 1`` with
    ``dot = c_prev . c_in`` yields the quadratic
    ``rho^2 + 2 * beta * dot * rho + (beta^2 - 1) = 0``,
    whose non-negative root is
    ``rho = sqrt(1 + beta^2 * (dot^2 - 1)) - beta * dot``.

    This generalises the orthonormal-input formula ``rho = sqrt(1 - beta^2)``
    from Howard & Kahana (2002, Eq. 3) and matches the update rule used by
    Polyn, Norman & Kahana (2009, Eq. 3) for non-orthogonal inputs. When
    ``dot = 0`` it reduces exactly to the orthonormal formula.
    """
    return float(
        np.sqrt(max(0.0, 1.0 + beta * beta * (dot_prev_in * dot_prev_in - 1.0)))
        - beta * dot_prev_in
    )


def update_global_context(
    c_prev: np.ndarray, beta_global: float, c_in: np.ndarray,
) -> np.ndarray:
    """c_G(t) = rho_G * c_G(t-1) + beta_G * c^IN_i.

    rho_G is chosen so that ||c_G(t)|| = 1 at every step, under the
    assumption that both ``c_prev`` and ``c_in`` are unit-norm. For strictly
    orthonormal inputs this reduces to rho = sqrt(1 - beta^2); for
    non-orthogonal unit-norm inputs (the usual case for multi-hot features
    that share active slots across items) the correction captures the
    c_prev . c_in cross-term. See ``_norm_preserving_rho``.
    """
    if not (0.0 < beta_global < 1.0):
        raise ValueError(f"beta_global must be in (0, 1); got {beta_global!r}")
    c_prev = np.asarray(c_prev, dtype=np.float64)
    c_in = np.asarray(c_in, dtype=np.float64)
    if c_prev.shape != c_in.shape:
        raise ValueError(
            f"update_global_context: shape mismatch {c_prev.shape} vs {c_in.shape}"
        )
    dot = float(np.dot(c_prev, c_in))
    rho = _norm_preserving_rho(beta_global, dot)
    return rho * c_prev + beta_global * c_in


def update_storyline_context(
    c_prev: np.ndarray, beta_storyline: float, c_in: np.ndarray,
) -> np.ndarray:
    """c_S(t) = rho_S * c_S(t_prev_S) + beta_S * c^IN_i.

    Callers MUST only invoke this when storyline S is active at the current
    step (i.e., the encoded event belongs to storyline S). Inactive storylines
    are frozen: the caller propagates the previous value unchanged.

    As with ``update_global_context``, rho_S is computed adaptively so that
    ||c_S(t)|| = 1 under non-orthogonal unit-norm inputs.
    """
    if not (0.0 < beta_storyline < 1.0):
        raise ValueError(f"beta_storyline must be in (0, 1); got {beta_storyline!r}")
    c_prev = np.asarray(c_prev, dtype=np.float64)
    c_in = np.asarray(c_in, dtype=np.float64)
    if c_prev.shape != c_in.shape:
        raise ValueError(
            f"update_storyline_context: shape mismatch {c_prev.shape} vs {c_in.shape}"
        )
    dot = float(np.dot(c_prev, c_in))
    rho = _norm_preserving_rho(beta_storyline, dot)
    return rho * c_prev + beta_storyline * c_in
