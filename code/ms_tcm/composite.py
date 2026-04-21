"""Composite encoding and retrieval contexts (sections 3.3-3.4).

Pure linear combinations; validation of w_G + w_S = 1 lives in
``ModelParameters``, not here (callers supply already-validated weights).
"""

from __future__ import annotations

import numpy as np


def compose_encoding(
    c_global: np.ndarray, c_storyline: np.ndarray,
    w_global: float, w_storyline: float,
) -> np.ndarray:
    """c_comp(i) = w_G * c_G(t) + w_S * c_S*(t)."""
    c_global = np.asarray(c_global, dtype=np.float64)
    c_storyline = np.asarray(c_storyline, dtype=np.float64)
    if c_global.shape != c_storyline.shape:
        raise ValueError(
            f"compose_encoding: shape mismatch {c_global.shape} vs {c_storyline.shape}"
        )
    return w_global * c_global + w_storyline * c_storyline


def compose_retrieval(
    c_global: np.ndarray, c_storyline: np.ndarray,
    w_global_ret: float, w_storyline_ret: float,
) -> np.ndarray:
    """c_ret(j) = w_G^ret * c_G(t_j) + w_S^ret * c_S_j(t_j)."""
    return compose_encoding(c_global, c_storyline, w_global_ret, w_storyline_ret)
