"""Tests for boundary-driven state transitions (T018 / US1 / v6 §2.2-2.4).

v6 Eq 4 (event boundary within storyline): c^item <- c^story
v6 Eq 5 (storyline switch): cache c^story_out to M^SC, then c^item <- c^story
v6 Eq 6 (storyline return): c^story <- lambda*c_tilde^story + (1-lambda)*c^story_prev
v6 Eq 7 (after return): c^item <- c^story
"""

from __future__ import annotations

import numpy as np
import pytest

from ms_tcm.boundaries import (
    apply_event_boundary,
    apply_storyline_return,
    apply_storyline_switch,
)
from ms_tcm.matrices import read_cached_story


def test_event_boundary_syncs_item_to_story() -> None:
    """v6 Eq 4: at an event boundary within a storyline, c^item snaps to c^story."""
    c_item = np.array([1.0, 0.0, 0.0])
    c_story = np.array([0.0, 0.6, 0.8])
    result = apply_event_boundary(c_item, c_story)
    np.testing.assert_array_almost_equal(result, c_story, decimal=12)
    # It should be a copy (not the same object) so callers can mutate freely.
    assert result is not c_story


def test_storyline_switch_caches_outgoing_and_syncs_item_to_incoming() -> None:
    """v6 Eq 5: at a switch, outgoing storyline is cached to M^SC and c^item
    synchronizes to the newly active storyline's context.
    """
    m_sc = np.zeros((3, 4))  # 3 storylines, d=4
    outgoing_idx = 0
    c_story_out = np.array([1.0, 0.0, 0.0, 0.0])
    c_story_new = np.array([0.0, 1.0, 0.0, 0.0])

    c_item_new, m_sc_updated = apply_storyline_switch(
        m_sc, outgoing_idx, c_story_out, c_story_new, n_storylines=3,
    )

    # c_item now equals the newly active storyline's context.
    np.testing.assert_array_almost_equal(c_item_new, c_story_new, decimal=12)

    # M^SC row 0 (the outgoing storyline) now holds c_story_out.
    np.testing.assert_array_almost_equal(m_sc_updated[0], c_story_out, decimal=12)


def test_storyline_return_with_lambda_one_fully_restores_cached_context() -> None:
    """With lambda=1, the storyline context is exactly the cached reinstatement."""
    m_sc = np.zeros((2, 3))
    # Pre-seed a cached context for storyline 0.
    cached = np.array([0.6, 0.8, 0.0])
    m_sc[0] = cached

    c_story_prev = np.array([0.0, 0.0, 1.0])  # current state for storyline 0
    c_story_new = apply_storyline_return(
        m_sc, returning_idx=0, c_story_prev=c_story_prev,
        lambda_reinstate=1.0, n_storylines=2,
    )
    np.testing.assert_array_almost_equal(c_story_new, cached, decimal=12)


def test_storyline_return_with_lambda_zero_behaves_like_no_reinstatement() -> None:
    """With lambda=0, the storyline context is unchanged (c_prev)."""
    m_sc = np.zeros((2, 3))
    m_sc[0] = np.array([0.6, 0.8, 0.0])  # cached

    c_story_prev = np.array([0.0, 0.0, 1.0])
    c_story_new = apply_storyline_return(
        m_sc, returning_idx=0, c_story_prev=c_story_prev,
        lambda_reinstate=0.0, n_storylines=2,
    )
    np.testing.assert_array_almost_equal(c_story_new, c_story_prev, decimal=12)


def test_storyline_return_blends_with_lambda_half() -> None:
    """v6 Eq 6 with lambda=0.5: c_story_new is the midpoint of cached and prev."""
    m_sc = np.zeros((2, 3))
    cached = np.array([1.0, 0.0, 0.0])
    m_sc[0] = cached
    c_story_prev = np.array([0.0, 1.0, 0.0])

    c_story_new = apply_storyline_return(
        m_sc, returning_idx=0, c_story_prev=c_story_prev,
        lambda_reinstate=0.5, n_storylines=2,
    )
    expected = 0.5 * cached + 0.5 * c_story_prev
    np.testing.assert_array_almost_equal(c_story_new, expected, decimal=12)


def test_storyline_return_with_never_cached_storyline_returns_prev() -> None:
    """If the returning storyline has no cached entry in M^SC (never switched out),
    the reinstatement is effectively the zero vector, so c_story_new = (1-lambda)*c_prev.
    """
    m_sc = np.zeros((2, 3))  # nothing cached
    c_story_prev = np.array([0.0, 1.0, 0.0])
    c_story_new = apply_storyline_return(
        m_sc, returning_idx=0, c_story_prev=c_story_prev,
        lambda_reinstate=0.8, n_storylines=2,
    )
    expected = 0.8 * np.zeros(3) + 0.2 * c_story_prev
    np.testing.assert_array_almost_equal(c_story_new, expected, decimal=12)


def test_storyline_return_rejects_out_of_range_lambda() -> None:
    m_sc = np.zeros((2, 3))
    with pytest.raises(ValueError, match="lambda_reinstate must be in"):
        apply_storyline_return(m_sc, 0, np.zeros(3), lambda_reinstate=1.1, n_storylines=2)


def test_storyline_switch_rejects_out_of_range_index() -> None:
    m_sc = np.zeros((2, 3))
    with pytest.raises(ValueError, match="outgoing_idx"):
        apply_storyline_switch(m_sc, 5, np.zeros(3), np.zeros(3), n_storylines=2)
