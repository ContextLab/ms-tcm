"""Tests for the CMR retrieval route (T019 / US1 / FR-005, FR-006).

v6 §3.1-3.2 + Cornell & Zhang 2025 Eqs 3, 7, 8-9:

- Activation: a_j = (M^IC)^T @ c^item_cue      (FR-005)
- Retrieval probabilities: p_j = softmax(k * a_j)  (FR-005)
- Within-trial retrieval drift: c^ret <- rho * c^ret + beta_rec * c^item_recalled  (FR-006, free-recall)
- Stopping rule: p_stop = exp(-epsilon_d * a^nr / a^r)  (FR-006, free-recall; C&Z Eq 7)
"""

from __future__ import annotations

import numpy as np
import pytest

from ms_tcm.retrieval import (
    activation,
    drift_retrieval_context,
    recall_probabilities,
    stopping_probability,
)


def test_activation_equals_MIC_T_times_c_item_cue() -> None:
    """v6 Eq 8 / CMR: a_j = (M^IC)^T @ c^item_cue."""
    # M^IC is (d, n_items); columns = per-item accumulated c^item vectors.
    m_ic = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ])  # d=3, n_items=2
    c_cue = np.array([1.0, 0.0, 0.0])  # cues item 0 exclusively
    a = activation(m_ic, c_cue)
    np.testing.assert_array_almost_equal(a, np.array([1.0, 0.0]), decimal=12)


def test_recall_probabilities_softmax_with_gain_k_peaks_on_highest_activation() -> None:
    """p_j = softmax(k * a_j): larger k sharpens toward the winner."""
    a = np.array([1.0, 0.5, 0.0])
    p_low_k = recall_probabilities(a, k=1.0)
    p_high_k = recall_probabilities(a, k=10.0)
    # Both sum to 1.
    assert abs(p_low_k.sum() - 1.0) < 1e-12
    assert abs(p_high_k.sum() - 1.0) < 1e-12
    # Higher k sharpens toward the argmax.
    assert p_high_k[0] > p_low_k[0]
    # Non-negative.
    assert np.all(p_low_k >= 0)
    assert np.all(p_high_k >= 0)


def test_recall_probabilities_mask_excludes_items() -> None:
    """A boolean mask excludes items (e.g. already-recalled) from competition."""
    a = np.array([1.0, 0.9, 0.5])
    mask = np.array([True, False, True])  # item 1 excluded
    p = recall_probabilities(a, k=5.0, mask=mask)
    assert p[1] == 0.0
    assert abs(p[0] + p[2] - 1.0) < 1e-12


def test_recall_probabilities_rejects_nan() -> None:
    """NaN activations must not silently produce NaN probabilities."""
    a = np.array([1.0, np.nan, 0.5])
    with pytest.raises(ValueError, match="NaN"):
        recall_probabilities(a, k=1.0)


def test_beta_rec_drifts_retrieval_toward_last_recalled() -> None:
    """CMR Eq 3: c^ret drifts toward c^item of the just-recalled item with rate beta_rec."""
    c_ret = np.array([1.0, 0.0, 0.0])
    c_item_recalled = np.array([0.0, 1.0, 0.0])
    c_ret_new = drift_retrieval_context(c_ret, beta_rec=0.5, c_item_recalled=c_item_recalled)
    # Unit norm preserved.
    assert abs(np.linalg.norm(c_ret_new) - 1.0) < 1e-12
    # Moved in the direction of the recalled item.
    rho = np.sqrt(1.0 - 0.5**2)
    expected = rho * c_ret + 0.5 * c_item_recalled
    np.testing.assert_array_almost_equal(c_ret_new, expected, decimal=12)


def test_stopping_probability_grows_as_all_recalled() -> None:
    """C&Z 2025 Eq 7: p_stop = exp(-epsilon_d * a^nr / a^r).

    When activation is concentrated in not-yet-recalled items, stopping
    probability is small. As more items get recalled (a^r grows relative to
    a^nr), stopping probability increases monotonically.
    """
    # Use moderately separated but not extreme ratios so the early-stage
    # stopping probability doesn't underflow to exactly 0. (The real model
    # uses balanced per-item activations rather than the 1000:1 ratio that
    # would hit float64 underflow.)
    p_early = stopping_probability(a_r_sum=1.0, a_nr_sum=3.0, epsilon_d=1.04)
    p_mid = stopping_probability(a_r_sum=1.0, a_nr_sum=1.0, epsilon_d=1.04)
    p_late = stopping_probability(a_r_sum=3.0, a_nr_sum=1.0, epsilon_d=1.04)
    assert 0.0 < p_early < p_mid < p_late <= 1.0


def test_stopping_probability_bounded_0_to_1() -> None:
    """p_stop ∈ [0, 1] for all valid inputs."""
    for a_r, a_nr in [(0.1, 10.0), (1.0, 1.0), (10.0, 0.0)]:
        p = stopping_probability(a_r_sum=a_r, a_nr_sum=a_nr, epsilon_d=1.04)
        assert 0.0 <= p <= 1.0


def test_stopping_probability_zero_a_r_handled_gracefully() -> None:
    """With a^r = 0 (nothing recalled yet, or no activation overlap), p_stop -> 0."""
    p = stopping_probability(a_r_sum=0.0, a_nr_sum=5.0, epsilon_d=1.04)
    assert p == 0.0
