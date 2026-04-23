"""The CMR retrieval route for v6 hierarchical CMR.

v6 §3 (instruction-blind retrieval) + Cornell & Zhang 2025 Eqs 3, 7, 8-9:

- ``activation``: a = (M^IC)^T @ c^item_cue
- ``recall_probabilities``: softmax(k * a), with optional mask for
  already-recalled items
- ``drift_retrieval_context``: within-trial retrieval context drift at rate
  beta_rec (free-recall only)
- ``stopping_probability``: per-attempt stopping rule (free-recall only;
  C&Z Eq 7)

Spec: specs/002-ms-tcm-v6-hcmr/spec.md FR-005, FR-006.
Data model: specs/002-ms-tcm-v6-hcmr/data-model.md §6.
"""

from __future__ import annotations

import numpy as np
from scipy.special import softmax as _softmax

from ms_tcm.drift import _norm_preserving_rho


def activation(m_ic: np.ndarray, c_item_cue: np.ndarray) -> np.ndarray:
    """v6 Eq 8: a_j = (M^IC)^T @ c^item_cue.

    ``m_ic`` is (d, n_items); ``c_item_cue`` is (d,). Returns a (n_items,)
    activation vector that drives the softmax retrieval readout.
    """
    m_ic = np.asarray(m_ic, dtype=np.float64)
    c_item_cue = np.asarray(c_item_cue, dtype=np.float64)
    if m_ic.shape[0] != c_item_cue.shape[0]:
        raise ValueError(
            f"activation: m_ic has {m_ic.shape[0]} rows but c_item_cue has "
            f"{c_item_cue.shape[0]} elements"
        )
    return m_ic.T @ c_item_cue


def recall_probabilities(
    a: np.ndarray, *, k: float, mask: np.ndarray | None = None,
) -> np.ndarray:
    """v6 Eq 9: P(j | cue) = softmax(k * a_j), optionally masked.

    ``mask`` is a boolean array of shape (n_items,) where False marks items
    to exclude from the competition (e.g. already-recalled items). Masked
    items get probability 0; the remaining items re-normalize to sum to 1.

    Guards against NaN activations (raises rather than silently propagating).
    """
    a = np.asarray(a, dtype=np.float64)
    if np.any(np.isnan(a)):
        raise ValueError("activation contains NaN; refusing to compute probabilities")
    if k <= 0:
        raise ValueError(f"k (softmax gain) must be > 0; got {k!r}")

    if mask is None:
        return _softmax(k * a)

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != a.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match activation shape {a.shape}"
        )
    if not mask.any():
        raise ValueError("mask excludes all items; cannot form a probability distribution")

    # Softmax over the masked subset; masked items get probability 0.
    scaled = k * a
    scaled_masked = np.where(mask, scaled, -np.inf)
    p = _softmax(scaled_masked)
    # Where the input was -inf the softmax returns 0; normalization is automatic.
    return p


def drift_retrieval_context(
    c_ret: np.ndarray, beta_rec: float, c_item_recalled: np.ndarray,
) -> np.ndarray:
    """C&Z 2025 Eq 3: within-trial retrieval context drift.

    c^ret_j = rho_rec * c^ret_{j-1} + beta_rec * c^item_recalled

    Used in free-recall between successive recalls: after recalling item i,
    the retrieval context shifts toward c^item_i so subsequent recalls are
    biased toward i's temporal neighbors (producing the lag-CRP signature).

    rho_rec is chosen so ||c^ret|| = 1 at every step under unit-norm inputs.
    """
    if not (0.0 < beta_rec < 1.0):
        raise ValueError(f"beta_rec must be in (0, 1); got {beta_rec!r}")
    c_ret = np.asarray(c_ret, dtype=np.float64)
    c_item_recalled = np.asarray(c_item_recalled, dtype=np.float64)
    if c_ret.shape != c_item_recalled.shape:
        raise ValueError(
            f"drift_retrieval_context: shape mismatch {c_ret.shape} vs {c_item_recalled.shape}"
        )
    dot = float(np.dot(c_ret, c_item_recalled))
    rho = _norm_preserving_rho(beta_rec, dot)
    return rho * c_ret + beta_rec * c_item_recalled


def stopping_probability(a_r_sum: float, a_nr_sum: float, epsilon_d: float) -> float:
    """C&Z 2025 Eq 7: per-attempt stopping probability.

    p_stop = exp(-epsilon_d * a^nr / a^r)

    Interpretation: when activation is concentrated on not-yet-recalled
    items (a^nr >> a^r), the ratio is large, the exponent is strongly
    negative, and p_stop is near 0 — the model keeps recalling. Late in the
    trial, when all activation is on already-recalled items (a^nr ~ 0),
    p_stop -> exp(0) = 1 and the model stops.

    Special case: if a_r_sum <= 0 the ratio is undefined; by convention we
    return p_stop = 0 (cannot justify stopping without any recalled-item
    activation). This matches C&Z's initialization: the stopping rule is
    only consulted after at least one recall.
    """
    if epsilon_d <= 0:
        raise ValueError(f"epsilon_d must be > 0; got {epsilon_d!r}")
    if a_r_sum <= 0.0:
        return 0.0
    ratio = a_nr_sum / a_r_sum
    return float(np.exp(-epsilon_d * ratio))
