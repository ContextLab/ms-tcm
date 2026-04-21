"""Similarity and recall-probability readout.

See ``specs/001-ms-tcm-impl/contracts/model-api.md`` §7 and research R4.
"""

from __future__ import annotations

import numpy as np
from scipy.special import softmax as _scipy_softmax


def cosine_similarity(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-30) -> float:
    """Cosine similarity with an epsilon guard.

    Returns 0.0 when either input has zero norm — never NaN.
    The epsilon is applied additively to each norm before division so that
    an exactly-zero vector (e.g. ``c_S(0)`` before any word has been encoded
    on that storyline) produces a finite 0.0 rather than ``0/0``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"cosine_similarity: shape mismatch {a.shape} vs {b.shape}"
        )
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / ((na + eps) * (nb + eps)))


def recall_probabilities(scores: np.ndarray) -> np.ndarray:
    """Softmax with temperature 1, numerically stable (log-sum-exp via scipy)."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(
            f"recall_probabilities expects a 1-D score vector; got shape {scores.shape}"
        )
    if np.all(np.isnan(scores)):
        # Propagate an all-NaN input so upstream callers can detect it; a normal
        # softmax would silently fold the NaN into the sum.
        return np.full_like(scores, np.nan)
    return _scipy_softmax(scores)
