"""Tests for associative-matrix updates (T014 / Phase 2).

Hand-derive small M^IC / M^SC / M^FC_exp increments and assert our in-place
updates match at 1e-12. Source: v6 Eqs 3, 5; CMR Eq 2a.
"""

from __future__ import annotations

import numpy as np

from ms_tcm.matrices import fc_exp_update, ic_update, read_cached_story, sc_update


def test_ic_update_hand_derived_one_step() -> None:
    """v6 Eq 3: M^IC += c_item · f_i^T for a 3-dim context, 2-item vocabulary."""
    m_ic = np.zeros((3, 2), dtype=np.float64)
    c_item = np.array([0.1, 0.2, 0.3])
    f_i = np.array([1.0, 0.0])  # item 0
    ic_update(m_ic, c_item, f_i)

    # Hand-derived: only column 0 populated, with c_item's values.
    expected = np.array([
        [0.1, 0.0],
        [0.2, 0.0],
        [0.3, 0.0],
    ])
    np.testing.assert_array_almost_equal(m_ic, expected, decimal=12)


def test_ic_update_accumulates_across_calls() -> None:
    """Two sequential updates on different items populate their own columns."""
    m_ic = np.zeros((2, 2), dtype=np.float64)
    ic_update(m_ic, np.array([1.0, 0.0]), np.array([1.0, 0.0]))   # item 0
    ic_update(m_ic, np.array([0.0, 1.0]), np.array([0.0, 1.0]))   # item 1
    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_array_almost_equal(m_ic, expected, decimal=12)


def test_ic_update_rejects_shape_mismatch() -> None:
    m_ic = np.zeros((3, 2))
    with __import__("pytest").raises(ValueError, match="expected m_ic shape"):
        ic_update(m_ic, np.array([0.0, 1.0]), np.array([1.0, 0.0]))  # c_item too short


def test_sc_update_hand_derived() -> None:
    """v6 Eq 5: ΔM^SC = g_s · c_story_out^T."""
    m_sc = np.zeros((4, 3), dtype=np.float64)  # 4 storylines, d=3
    storyline_onehot = np.array([0.0, 1.0, 0.0, 0.0])  # storyline 1
    c_story_out = np.array([0.5, 0.5, 0.0])
    sc_update(m_sc, storyline_onehot, c_story_out)

    expected = np.zeros((4, 3))
    expected[1] = c_story_out
    np.testing.assert_array_almost_equal(m_sc, expected, decimal=12)


def test_fc_exp_update_hand_derived() -> None:
    """CMR Eq 2a: ΔM^FC_exp = c_in · f_i^T."""
    m_fc_exp = np.zeros((2, 3))  # d=2, 3 items
    c_in = np.array([0.3, 0.7])
    f_i = np.array([0.0, 1.0, 0.0])  # item 1
    fc_exp_update(m_fc_exp, c_in, f_i)

    expected = np.array([
        [0.0, 0.3, 0.0],
        [0.0, 0.7, 0.0],
    ])
    np.testing.assert_array_almost_equal(m_fc_exp, expected, decimal=12)


def test_read_cached_story_reproduces_input_at_1e_minus_12() -> None:
    """Round-trip: write a cached context for storyline 2, read it back."""
    m_sc = np.zeros((3, 4), dtype=np.float64)  # 3 storylines, d=4
    onehot_2 = np.array([0.0, 0.0, 1.0])
    cached = np.array([0.25, 0.25, 0.25, 0.25])
    sc_update(m_sc, onehot_2, cached)

    readback = read_cached_story(m_sc, onehot_2)
    np.testing.assert_array_almost_equal(readback, cached, decimal=12)


def test_read_cached_story_sums_when_storyline_returns_multiple_times() -> None:
    """Multiple caches to the same storyline accumulate (outer-product sum)."""
    m_sc = np.zeros((2, 3))
    onehot = np.array([1.0, 0.0])
    sc_update(m_sc, onehot, np.array([1.0, 0.0, 0.0]))
    sc_update(m_sc, onehot, np.array([0.0, 1.0, 0.0]))
    readback = read_cached_story(m_sc, onehot)
    np.testing.assert_array_almost_equal(
        readback, np.array([1.0, 1.0, 0.0]), decimal=12,
    )


def test_read_cached_story_returns_zero_for_storyline_never_cached() -> None:
    """A storyline that hasn't switched out yet has all-zero cached context."""
    m_sc = np.zeros((2, 3))
    readback = read_cached_story(m_sc, np.array([0.0, 1.0]))
    np.testing.assert_array_almost_equal(readback, np.zeros(3), decimal=12)
