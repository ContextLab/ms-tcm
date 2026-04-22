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


def recall_probabilities(scores: np.ndarray, *, tau: float = 1.0) -> np.ndarray:
    """Softmax ``P(i) = exp(tau * s_i) / sum_j exp(tau * s_j)``.

    ``tau`` is an inverse-temperature / gain parameter. At ``tau = 1`` the
    output is the straight softmax of the raw scores; higher ``tau`` sharpens
    the distribution toward the winning candidate. This is a practical
    necessity when scores are bounded cosine similarities in [-1, 1]: with
    ``tau = 1`` the softmax of a distinguishable-but-bounded similarity
    vector is close to uniform, which erases the contiguity and recency
    signatures that TCM is supposed to capture. TCM-A (Sederberg et al.,
    2008) implements the same sharpening through a separate accumulator
    dynamics; fitting a single ``tau`` is a lower-fidelity approximation
    that nonetheless recovers the qualitative shape of SPC, P(first recall),
    and lag-CRP.

    Numerically stable via scipy's log-sum-exp softmax.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(
            f"recall_probabilities expects a 1-D score vector; got shape {scores.shape}"
        )
    if np.all(np.isnan(scores)):
        # Propagate an all-NaN input so upstream callers can detect it; a normal
        # softmax would silently fold the NaN into the sum.
        return np.full_like(scores, np.nan)
    if tau != 1.0:
        scores = float(tau) * scores
    return _scipy_softmax(scores)
