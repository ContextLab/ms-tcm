"""Item-level and storyline-level context drift (v6 §2.1).

Pure functions — all list-boundary + boundary-synchronization bookkeeping
is the caller's responsibility (see ``ms_tcm.boundaries``); the orchestrator
in ``ms_tcm.hcmr`` weaves drift and boundary logic together.

v6 Eq 1 (item-level):    c^item_i = rho_enc   * c^item_{i-1}   + beta_enc   * c^IN_i
v6 Eq 2 (storyline):     c^story_i = rho_story * c^story_{j-1} + beta_story * c^IN_i

where rho is chosen so ||c(t)|| = 1 given unit-norm inputs. For orthonormal
inputs this reduces to rho = sqrt(1 - beta^2); for non-orthogonal inputs
the generalized formula handles the cross term (Howard & Kahana 2002 /
Polyn et al. 2009 Eq 3 / CMR Eq 1).
"""

from __future__ import annotations

import numpy as np


def _norm_preserving_rho(beta: float, dot_prev_in: float) -> float:
    """Generalized rho so that ||rho*c_prev + beta*c_in|| = 1 for unit-norm c_prev, c_in.

    Derivation: expand ||rho*c_prev + beta*c_in||^2 = 1 with
    dot = c_prev . c_in to get rho^2 + 2*beta*dot*rho + (beta^2 - 1) = 0.
    The non-negative root is rho = sqrt(1 + beta^2*(dot^2 - 1)) - beta*dot.
    When dot = 0 this reduces to rho = sqrt(1 - beta^2).
    """
    return float(
        np.sqrt(max(0.0, 1.0 + beta * beta * (dot_prev_in * dot_prev_in - 1.0)))
        - beta * dot_prev_in
    )


def update_item_context(c_prev: np.ndarray, beta_enc: float, c_in: np.ndarray) -> np.ndarray:
    """v6 §2.1 Eq 1: c^item_i = rho_enc * c^item_{i-1} + beta_enc * c^IN_i.

    rho_enc is chosen so ||c^item|| = 1 at every step under unit-norm
    inputs. Callers MUST supply unit-norm c_prev and c_in; the function
    does not enforce this (doing so would change cost-sensitive inner-loop
    behavior) but the `test_drift.py::test_*_preserves_unit_norm` tests
    verify it under the documented assumption.
    """
    if not (0.0 < beta_enc < 1.0):
        raise ValueError(f"beta_enc must be in (0, 1); got {beta_enc!r}")
    c_prev = np.asarray(c_prev, dtype=np.float64)
    c_in = np.asarray(c_in, dtype=np.float64)
    if c_prev.shape != c_in.shape:
        raise ValueError(
            f"update_item_context: shape mismatch {c_prev.shape} vs {c_in.shape}"
        )
    dot = float(np.dot(c_prev, c_in))
    rho = _norm_preserving_rho(beta_enc, dot)
    return rho * c_prev + beta_enc * c_in


def update_story_context(c_prev: np.ndarray, beta_story: float, c_in: np.ndarray) -> np.ndarray:
    """v6 §2.1 Eq 2: c^story_i = rho_story * c^story_{j-1} + beta_story * c^IN_i.

    Callers MUST only invoke this when the current item belongs to this
    storyline (i.e., the storyline is "active"). Inactive storylines are
    carried forward unchanged — the orchestrator handles that.
    """
    if not (0.0 < beta_story < 1.0):
        raise ValueError(f"beta_story must be in (0, 1); got {beta_story!r}")
    c_prev = np.asarray(c_prev, dtype=np.float64)
    c_in = np.asarray(c_in, dtype=np.float64)
    if c_prev.shape != c_in.shape:
        raise ValueError(
            f"update_story_context: shape mismatch {c_prev.shape} vs {c_in.shape}"
        )
    dot = float(np.dot(c_prev, c_in))
    rho = _norm_preserving_rho(beta_story, dot)
    return rho * c_prev + beta_story * c_in
