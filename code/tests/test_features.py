"""Tests for vectorized encode_features + DatasetFeatureCache (T015, T016).

FR-030 Tier 1: ``encode_features`` vectorized with no pandas.iterrows().
``DatasetFeatureCache`` caches per-Dataset so likelihood evaluations skip
re-encoding.

Bit-identity guarantee: the vectorized encoder produces exactly the same
output as the legacy iterrows encoder on the bundled FRFR-category dataset.
"""

from __future__ import annotations

import numpy as np

from ms_tcm.dataset import load_dataset
from ms_tcm.features import (
    DatasetFeatureCache,
    FEATURE_DIM,
    _encode_features_iterrows,
    encode_features,
)
from ms_tcm.frfr import load_frfr_category


def test_vectorization_bit_identity_on_frfr_category() -> None:
    """The vectorized encoder produces bit-identical output to the iterrows legacy encoder.

    This is the core FR-030 correctness gate: no silent regression allowed
    during the Tier 1 performance optimization.
    """
    ds = load_frfr_category()
    vec = encode_features(ds.presented)
    leg = _encode_features_iterrows(ds.presented)
    assert vec.shape == leg.shape == (7680, FEATURE_DIM)
    # Bit-identical (not approximate): both paths perform the same
    # arithmetic in the same order modulo the vectorized scatter.
    np.testing.assert_array_equal(vec, leg)


def test_vectorization_bit_identity_normalize_false() -> None:
    """Bit-identity holds for the un-normalized path too."""
    ds = load_frfr_category()
    vec = encode_features(ds.presented, normalize=False)
    leg = _encode_features_iterrows(ds.presented, normalize=False)
    np.testing.assert_array_equal(vec, leg)


def test_unknown_category_routes_to_unknown_slot() -> None:
    """Unrecognized category values land in the reserved UNKNOWN slot."""
    import pandas as pd

    # Construct a 1-row DataFrame with an out-of-vocabulary category.
    row = {
        "category": "UNKNOWN_CATEGORY",
        "size": "small",
        "first_letter": "A",
        "word_length": 5,
        "color_r": 0,
        "color_g": 0,
        "color_b": 0,
        "pos_x": 0.0,
        "pos_y": 0.0,
    }
    df = pd.DataFrame([row])
    vec = encode_features(df, normalize=False)
    leg = _encode_features_iterrows(df, normalize=False)
    np.testing.assert_array_equal(vec, leg)


def test_dataset_feature_cache_hits_after_first_call() -> None:
    """Second call with the same Dataset + pre_matrix returns cached array (same object)."""
    ds = load_frfr_category()
    cache = DatasetFeatureCache()
    pre = object()
    first = cache.get_or_compute(ds, pre)
    second = cache.get_or_compute(ds, pre)
    assert first is second  # identity, not equality — proves it's cached


def test_dataset_feature_cache_recomputes_when_pre_matrix_changes() -> None:
    """Different pre_matrix object triggers recomputation."""
    ds = load_frfr_category()
    cache = DatasetFeatureCache()
    first = cache.get_or_compute(ds, object())
    second = cache.get_or_compute(ds, object())  # new pre_matrix
    assert first is not second
    np.testing.assert_array_equal(first, second)  # values still identical
