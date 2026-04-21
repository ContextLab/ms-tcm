"""Optional Section-5 mechanisms: resumption reinstatement, conversational
references, differential interference. Each function is a no-op when its
respective parameter is 0 / empty (FR-007).
"""

from __future__ import annotations

import numpy as np


def apply_resumption_reinstatement(
    c_storyline_prev: np.ndarray, gamma: float,
) -> np.ndarray:
    """Return gamma * c_S(t_prev^S), the reinstatement boost added to the
    storyline-context update at a storyline resumption (Section 5.1).
    """
    if gamma == 0.0:
        return np.zeros_like(c_storyline_prev, dtype=np.float64)
    return float(gamma) * np.asarray(c_storyline_prev, dtype=np.float64)


def apply_conversational_references(
    ref_inputs: list[np.ndarray], alphas: list[float],
) -> np.ndarray | None:
    """Sum of alpha_k * c^IN_{ref_k} across a list of referenced events
    (Section 5.2). Returns None when no references are provided.
    """
    if not ref_inputs:
        return None
    if len(ref_inputs) != len(alphas):
        raise ValueError(
            f"ref_inputs and alphas must have the same length; "
            f"got {len(ref_inputs)} vs {len(alphas)}"
        )
    out = np.zeros_like(np.asarray(ref_inputs[0], dtype=np.float64))
    for r, a in zip(ref_inputs, alphas):
        out = out + float(a) * np.asarray(r, dtype=np.float64)
    return out


def interference_factor(lambda_interference: float, i_ij: float) -> float:
    """exp(-lambda * I_{ij}); equals 1.0 when lambda is 0 (no interference)."""
    if lambda_interference == 0.0:
        return 1.0
    return float(np.exp(-lambda_interference * i_ij))
