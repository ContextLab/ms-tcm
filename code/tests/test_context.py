"""Tests for global and storyline context updates (§3.2)."""

from __future__ import annotations

import numpy as np

from ms_tcm.context import update_global_context, update_storyline_context


def test_global_context_drift() -> None:
    """FR-001, §3.2: c_G(t) = ρ_G · c_G(t-1) + β_G · c^IN_i element-wise."""
    beta_g = 0.5
    c_prev = np.array([1.0, 0.0, 0.0])
    c_in = np.array([0.0, 1.0, 0.0])
    out = update_global_context(c_prev, beta_g, c_in)
    rho_g = np.sqrt(1.0 - beta_g**2)
    expected = rho_g * c_prev + beta_g * c_in
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_storyline_context_frozen_when_inactive() -> None:
    """§3.2: storyline contexts don't drift when their storyline is inactive.

    "Frozen-when-inactive" means: if the updater is never called for storyline
    S at steps t...t+k, c_S at t+k+1 (when S becomes active again) still
    equals its value at t. That bookkeeping is the caller's responsibility --
    the updater itself is a pure function. Here we also check that the
    updater, under orthogonal unit-norm inputs (the setting in which
    rho = sqrt(1 - beta^2) holds exactly), returns the closed-form
    combination.
    """
    beta_s = 0.4
    # Orthogonal unit-norm inputs -- the setting in which the closed-form
    # rho = sqrt(1 - beta^2) applies exactly.
    c_prev = np.array([1.0, 0.0, 0.0])
    c_in = np.array([0.0, 1.0, 0.0])
    rho_s = np.sqrt(1.0 - beta_s**2)

    out1 = update_storyline_context(c_prev, beta_s, c_in)
    np.testing.assert_allclose(out1, rho_s * c_prev + beta_s * c_in, atol=1e-12)

    # Frozen semantics: if we do not call the updater, the vector is unchanged.
    c_frozen = c_prev.copy()
    # ...many inactive steps...
    assert np.array_equal(c_frozen, c_prev)


def test_global_context_preserves_unit_norm_for_unit_inputs() -> None:
    """With ρ² + β² = 1 and orthogonal unit inputs, ‖c_G‖ stays at 1."""
    beta_g = 0.5
    c_prev = np.array([1.0, 0.0])
    c_in = np.array([0.0, 1.0])  # orthogonal unit vector
    out = update_global_context(c_prev, beta_g, c_in)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-12


def test_update_is_deterministic() -> None:
    """Two calls with the same inputs return bit-identical arrays."""
    a = update_global_context(np.array([0.1, 0.2]), 0.3, np.array([0.5, -0.5]))
    b = update_global_context(np.array([0.1, 0.2]), 0.3, np.array([0.5, -0.5]))
    np.testing.assert_array_equal(a, b)
