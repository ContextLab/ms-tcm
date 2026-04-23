"""Tests for ModelParameters v6 defaults (T010 / US5 / FR-009).

Each default MUST match Cornell & Zhang 2025 Table 1 (or v6 §5 for λ) to
exact equality, and each field's docstring MUST cite the source.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from ms_tcm.params import ModelParameters


def test_default_parameter_values_match_cornell_zhang_2025_table_1_and_v6_section_5() -> None:
    """FR-009: defaults match C&Z 2025 Table 1 + v6 §5 to exact equality."""
    p = ModelParameters()

    # Cornell & Zhang 2025 Table 1 (page-level anchor in CornZhan25.pdf):
    assert p.beta_enc == 0.679,        "beta_enc must match C&Z 2025 Table 1"
    assert p.beta_story == 0.400,      "beta_story must match C&Z 2025 Table 1 (β_list)"
    assert p.gamma_fc == 0.315,        "gamma_fc must match C&Z 2025 Table 1"
    assert p.k == 6.50,                "k must match C&Z 2025 Table 1"
    assert p.beta_rec == 0.326,        "beta_rec must match C&Z 2025 Table 1"
    assert p.epsilon_d == 1.04,        "epsilon_d must match C&Z 2025 Table 1"

    # v6 §5:
    assert p.lambda_reinstate == 0.80, "lambda_reinstate must match v6 §5"


def test_each_parameter_field_docstring_cites_its_source() -> None:
    """FR-009: every field's docstring mentions the source page or section.

    A runtime-readable docstring per field is carried via the class body (we
    attach the docstring text to ``__doc__`` by convention; dataclass fields
    don't carry per-field docstrings in the standard ``dataclasses`` module,
    so we parse the class body source instead).
    """
    # Use the module's source rather than field metadata because dataclass
    # fields don't natively carry per-field docstrings.
    import inspect
    src = inspect.getsource(ModelParameters)

    # Each field must cite AT LEAST ONE of the listed substrings in the
    # docstring block that immediately follows its declaration. Accepting
    # either "v6 §5" or "two_level_cmr_v6.pdf §5" covers both abbreviated
    # and full-path citation styles.
    required_any_of_by_field: dict[str, list[str]] = {
        "beta_enc":         ["Cornell & Zhang 2025 Table 1"],
        "beta_story":       ["Cornell & Zhang 2025 Table 1"],
        "gamma_fc":         ["Cornell & Zhang 2025 Table 1"],
        "k":                ["Cornell & Zhang 2025 Table 1"],
        "lambda_reinstate": ["v6 §5", "two_level_cmr_v6.pdf §5"],
        "beta_rec":         ["Cornell & Zhang 2025 Table 1"],
        "epsilon_d":        ["Cornell & Zhang 2025 Table 1"],
    }
    for name, needles in required_any_of_by_field.items():
        idx = src.find(f"{name}:")
        assert idx >= 0, f"field {name} not found in ModelParameters source"
        # Bound the window by the NEXT top-level field declaration, not a
        # fixed character count — docstrings vary in length.
        next_field_idx = len(src)
        for other in required_any_of_by_field:
            if other == name:
                continue
            j = src.find(f"    {other}:", idx + 1)
            if 0 <= j < next_field_idx:
                next_field_idx = j
        window = src[idx:next_field_idx]
        found = any(needle in window for needle in needles)
        assert found, (
            f"field {name!r} docstring must cite one of {needles!r}; "
            f"none found in the field's docstring block"
        )


def test_beta_enc_greater_than_beta_story_enforced() -> None:
    """FR-001 / v6 §2.1: item-level drift must exceed storyline-level drift."""
    with pytest.raises(ValueError, match="beta_enc > beta_story"):
        ModelParameters(beta_enc=0.3, beta_story=0.4)


def test_standard_tcm_reduction_sets_lambda_zero_and_standard_tcm_true() -> None:
    """FR-011: the reduction explicitly forces both markers."""
    p = ModelParameters.standard_tcm_reduction()
    assert p.lambda_reinstate == 0.0
    assert p.standard_tcm is True


def test_standard_tcm_reduction_rejects_conflicting_lambda() -> None:
    """FR-011: passing a non-zero lambda to the reduction raises."""
    with pytest.raises(ValueError):
        ModelParameters.standard_tcm_reduction(lambda_reinstate=0.5)


def test_replace_revalidates() -> None:
    """``ModelParameters.replace()`` must re-run __post_init__ validation."""
    p = ModelParameters()
    with pytest.raises(ValueError):
        # Replacement that violates beta_enc > beta_story.
        p.replace(beta_enc=0.1, beta_story=0.2)


def test_paradigm_literal_rejects_invalid() -> None:
    """Invalid paradigm strings raise."""
    with pytest.raises(ValueError, match="paradigm must be"):
        ModelParameters(paradigm="cued-recall")  # type: ignore[arg-type]


def test_cued_recall_paradigm_relaxes_free_recall_only_bounds() -> None:
    """A7 resolution: cued-recall paradigm ignores β_rec / ε_d bounds."""
    # In cued recall, β_rec and ε_d are unused; the constructor should accept
    # default values (free-recall-valid) without complaint.
    p = ModelParameters(paradigm="cued_recall")
    assert p.paradigm == "cued_recall"
