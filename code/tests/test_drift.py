"""Tests for item-level and storyline-level drift (T017 / US1 / v6 §2.1).

v6 §2.1 Eqs 1-2: item-level and storyline-level contexts drift toward the
current input c^IN:

    c^item_i = rho_enc * c^item_{i-1} + beta_enc * c^IN_i
    c^story_i = rho_story * c^story_{i-1} + beta_story * c^IN_i

rho values are chosen internally so ||c|| = 1 under unit-norm inputs
(FR-008). For orthonormal inputs rho = sqrt(1 - beta^2); for non-orthogonal
but unit-norm inputs a correction term handles the cross term.
"""

from __future__ import annotations

import numpy as np
import pytest

from ms_tcm.drift import update_item_context, update_story_context


def _e_start(dim: int = 4) -> np.ndarray:
    """Reserved list-start unit vector (one-hot at index 0)."""
    v = np.zeros(dim)
    v[0] = 1.0
    return v


def _e(dim: int, i: int) -> np.ndarray:
    """One-hot unit vector with 1.0 at index i."""
    v = np.zeros(dim)
    v[i] = 1.0
    return v


def test_item_drift_preserves_unit_norm_orthonormal() -> None:
    """With orthonormal unit-norm inputs, ||c^item|| = 1 at every step."""
    c = _e_start(8)
    for i in range(1, 5):
        c = update_item_context(c, beta_enc=0.679, c_in=_e(8, i))
        assert abs(np.linalg.norm(c) - 1.0) < 1e-12


def test_story_drift_preserves_unit_norm_orthonormal() -> None:
    c = _e_start(8)
    for i in range(1, 5):
        c = update_story_context(c, beta_story=0.400, c_in=_e(8, i))
        assert abs(np.linalg.norm(c) - 1.0) < 1e-12


def test_item_drift_one_step_matches_hand_derivation() -> None:
    """v6 Eq 1: c^item_1 = rho_enc * e_start + beta_enc * c^IN_1, with
    rho_enc = sqrt(1 - beta_enc^2) under orthonormal inputs.
    """
    beta_enc = 0.679
    c_prev = _e_start(4)
    c_in = _e(4, 1)  # orthogonal to e_start
    c_new = update_item_context(c_prev, beta_enc, c_in)

    rho = np.sqrt(1.0 - beta_enc**2)
    expected = rho * c_prev + beta_enc * c_in
    np.testing.assert_array_almost_equal(c_new, expected, decimal=12)
    # Unit norm.
    assert abs(np.linalg.norm(c_new) - 1.0) < 1e-12


def test_story_drift_one_step_matches_hand_derivation() -> None:
    beta_story = 0.400
    c_prev = _e_start(4)
    c_in = _e(4, 1)
    c_new = update_story_context(c_prev, beta_story, c_in)

    rho = np.sqrt(1.0 - beta_story**2)
    expected = rho * c_prev + beta_story * c_in
    np.testing.assert_array_almost_equal(c_new, expected, decimal=12)


def test_item_drift_non_orthogonal_input_still_unit_norm() -> None:
    """Non-orthogonal unit-norm inputs: rho compensates for the cross term
    so ||c^item|| remains 1. Verifies the generalized rho formula from
    Howard & Kahana (2002) / Polyn et al. (2009) Eq 3.
    """
    # c_prev and c_in both unit-norm but not orthogonal.
    c_prev = np.array([0.6, 0.8, 0.0, 0.0])
    c_in = np.array([0.8, 0.6, 0.0, 0.0])
    assert abs(np.linalg.norm(c_prev) - 1.0) < 1e-12
    assert abs(np.linalg.norm(c_in) - 1.0) < 1e-12
    c_new = update_item_context(c_prev, beta_enc=0.5, c_in=c_in)
    assert abs(np.linalg.norm(c_new) - 1.0) < 1e-10


def test_item_drift_rejects_out_of_range_beta() -> None:
    with pytest.raises(ValueError, match="beta_enc must be in"):
        update_item_context(_e_start(4), beta_enc=0.0, c_in=_e(4, 1))
    with pytest.raises(ValueError, match="beta_enc must be in"):
        update_item_context(_e_start(4), beta_enc=1.0, c_in=_e(4, 1))


def test_story_drift_rejects_out_of_range_beta() -> None:
    with pytest.raises(ValueError, match="beta_story must be in"):
        update_story_context(_e_start(4), beta_story=0.0, c_in=_e(4, 1))


def test_drift_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        update_item_context(_e_start(4), beta_enc=0.5, c_in=_e(5, 1))
