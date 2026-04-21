"""Tests for cosine similarity and the softmax recall-probability readout."""

from __future__ import annotations

import numpy as np

from ms_tcm import cosine_similarity, recall_probabilities


def test_cosine_similarity_unit_vectors() -> None:
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == -1.0
    assert (
        abs(cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-12
    )


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    """FR-005 + research R4: zero-norm input yields 0.0, never NaN."""
    z = np.zeros(3)
    assert cosine_similarity(z, np.array([1.0, 2.0, 3.0])) == 0.0
    assert cosine_similarity(np.array([1.0, 2.0, 3.0]), z) == 0.0
    assert cosine_similarity(z, z) == 0.0


def test_recall_probabilities_sum_to_one() -> None:
    """FR-005: softmax output sums to 1 within 1e-12."""
    rng = np.random.default_rng(0)
    scores = rng.normal(size=20)
    p = recall_probabilities(scores)
    assert abs(p.sum() - 1.0) < 1e-12
    assert np.all(p >= 0.0)


def test_recall_probabilities_extreme_inputs_stable() -> None:
    """Softmax is numerically stable for extreme scores (log-sum-exp trick)."""
    scores = np.array([1000.0, 1001.0, 999.0])
    p = recall_probabilities(scores)
    assert abs(p.sum() - 1.0) < 1e-12
    assert not np.any(np.isnan(p))


def test_recall_probabilities_all_nan_propagates() -> None:
    """All-NaN input returns NaN (caller has upstream numerical error)."""
    out = recall_probabilities(np.array([np.nan, np.nan, np.nan]))
    assert np.all(np.isnan(out))
