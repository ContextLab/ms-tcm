"""Tests for the composite encoding/retrieval context math (sections 3.3-3.4, 4.4)."""

from __future__ import annotations

import numpy as np
import pytest

from ms_tcm import ModelParameters
from ms_tcm.composite import compose_encoding, compose_retrieval
from ms_tcm.context import update_global_context, update_storyline_context


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_section_4_4_numerical_anchor() -> None:
    """notes/ms-tcm.pdf section 4.4: at beta_G = beta_S = 0.5, w_G = 0.2, w_S = 0.8,
    m = 3, the composite-similarity quantity is sqrt(0.75) = 0.8660254... (grouped)
    and 0.2 * 0.75^2 + 0.8 * sqrt(0.75) = 0.8053203... (bridge). The spec rounds
    these to 0.866 and 0.806 at three decimals; our assertion uses the exact
    closed-form values at 1e-6 tolerance. Pins FR-013a, SC-002.

    Note on quantity definition: the spec defines "composite similarity" as the
    linear combination w_G * sim(c_G(i), c_G(i+1)) + w_S * sim(c_S(i), c_S(i+1)),
    NOT as cos(c_comp(i), c_comp(i+1)). The two coincide only when c_G(i) = c_S(i)
    and c_G(i+1) = c_S(i+1), which is true in the grouped case (shared drift
    history) but not in the bridge case (where c_G has drifted through m gap
    inputs that c_S has not seen). The weighted-sum quantity is what drives the
    model's scientific argument in section 4.3, so it is what we test here.
    """
    beta_g = beta_s = 0.5
    w_g, w_s = 0.2, 0.8

    # Feature space: one reserved list-start dimension + mutually orthogonal
    # word directions. Mutual orthogonality is required for the section-4.2
    # derivation to hold (per-stream similarities reduce to ρ^k in closed form).
    eye = np.eye(7)
    e_start = eye[0]
    f_i = eye[1]
    f_next = eye[2]
    f_gap_1 = eye[3]
    f_gap_2 = eye[4]
    f_gap_3 = eye[5]

    # Initialize both streams to the list-start unit vector.
    c_g_init = e_start.copy()
    c_s_init = e_start.copy()

    # Encode event i at t=1 on storyline S.
    c_g_i = update_global_context(c_g_init, beta_g, f_i)
    c_s_i = update_storyline_context(c_s_init, beta_s, f_i)

    # --- Grouped case: event i+1 is the next step in storyline S.
    c_g_next_grouped = update_global_context(c_g_i, beta_g, f_next)
    c_s_next_grouped = update_storyline_context(c_s_i, beta_s, f_next)
    sim_g_grouped = _cosine(c_g_i, c_g_next_grouped)
    sim_s_grouped = _cosine(c_s_i, c_s_next_grouped)
    composite_sim_grouped = w_g * sim_g_grouped + w_s * sim_s_grouped
    expected_grouped = np.sqrt(0.75)  # = 0.8660254...; spec rounds to 0.866.
    assert abs(composite_sim_grouped - expected_grouped) < 1e-6, (
        f"grouped composite similarity {composite_sim_grouped!r}; "
        f"expected {expected_grouped!r}"
    )

    # --- Bridge case: m=3 intervening events from other storylines drift the
    # global stream; storyline S sees only one drift step between i and i+1.
    c_g_after_gap = c_g_i.copy()
    for f_gap in (f_gap_1, f_gap_2, f_gap_3):
        c_g_after_gap = update_global_context(c_g_after_gap, beta_g, f_gap)
    c_g_next_bridge = update_global_context(c_g_after_gap, beta_g, f_next)
    c_s_next_bridge = update_storyline_context(c_s_i, beta_s, f_next)
    sim_g_bridge = _cosine(c_g_i, c_g_next_bridge)
    sim_s_bridge = _cosine(c_s_i, c_s_next_bridge)
    composite_sim_bridge = w_g * sim_g_bridge + w_s * sim_s_bridge
    # rho^4 = 0.75^2 = 0.5625; rho = sqrt(0.75); so expected =
    # 0.2 * 0.5625 + 0.8 * sqrt(0.75) = 0.1125 + 0.6928203... = 0.8053203...
    expected_bridge = 0.2 * 0.5625 + 0.8 * np.sqrt(0.75)
    assert abs(composite_sim_bridge - expected_bridge) < 1e-6, (
        f"bridge composite similarity {composite_sim_bridge!r}; "
        f"expected {expected_bridge!r}"
    )

    # Per-stream sanity checks: these are the ρ^k identities from section 4.2.
    rho = np.sqrt(1.0 - beta_g**2)
    assert abs(sim_g_grouped - rho) < 1e-12
    assert abs(sim_s_grouped - rho) < 1e-12
    assert abs(sim_g_bridge - rho**4) < 1e-12     # m+1 = 4 global drift steps
    assert abs(sim_s_bridge - rho) < 1e-12         # only one storyline drift step


def test_composite_weights_sum_to_one_enforced() -> None:
    """FR-003: ModelParameters rejects w_G + w_S != 1 within 1e-12."""
    ModelParameters(w_global=0.2, w_storyline=0.8)  # ok
    with pytest.raises(ValueError, match="sum to 1"):
        ModelParameters(w_global=0.3, w_storyline=0.5)
    with pytest.raises(ValueError, match="sum to 1"):
        ModelParameters(w_global=0.1, w_storyline=0.7)


def test_retrieval_weights_independent_of_encoding() -> None:
    """FR-004: retrieval weights can differ from encoding weights."""
    p = ModelParameters(
        w_global=0.2, w_storyline=0.8,
        w_global_ret=0.5, w_storyline_ret=0.5,
    )
    assert p.w_global == 0.2
    assert p.w_global_ret == 0.5
    # Retrieval weights still must sum to 1.
    with pytest.raises(ValueError, match="retrieval weights"):
        ModelParameters(
            w_global=0.2, w_storyline=0.8,
            w_global_ret=0.3, w_storyline_ret=0.5,
        )


def test_compose_encoding_is_linear() -> None:
    """c_comp = w_G * c_G + w_S * c_S (data-model.md section 4.1 step 4)."""
    c_g = np.array([1.0, 0.0, 0.0])
    c_s = np.array([0.0, 1.0, 0.0])
    out = compose_encoding(c_g, c_s, 0.2, 0.8)
    np.testing.assert_allclose(out, np.array([0.2, 0.8, 0.0]), atol=1e-12)


def test_compose_retrieval_uses_retrieval_weights() -> None:
    """Retrieval composite uses w_G^ret / w_S^ret, not the encoding weights."""
    c_g = np.array([1.0, 0.0])
    c_s = np.array([0.0, 1.0])
    out = compose_retrieval(c_g, c_s, 0.7, 0.3)
    np.testing.assert_allclose(out, np.array([0.7, 0.3]), atol=1e-12)
