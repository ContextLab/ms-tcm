"""Tests for MFCPreMatrix / IdentityPreMatrix / EmbeddingPreMatrix (T012 / US4 / FR-007).

v6 §1.5: M^FC_pre is the pre-experimental item-to-context matrix. Identity
default, optional pre-trained embeddings hook.
"""

from __future__ import annotations

import numpy as np
import pytest

from ms_tcm.preexp import (
    EmbeddingPreMatrix,
    IdentityPreMatrix,
    MFCPreMatrix,
)


def test_identity_preexp_returns_onehot_rows() -> None:
    """IdentityPreMatrix.apply returns a one-hot row per item index.

    v6 §1.5 Option 1: no semantic overlap — each item's pre-experimental
    context is orthogonal.
    """
    pre = IdentityPreMatrix(n_items=15, n_features=71)
    out = pre.apply(np.array([3]))
    assert out.shape == (1, 71)
    assert out[0, 3] == 1.0
    # All other columns zero.
    mask = np.ones(71, dtype=bool)
    mask[3] = False
    assert np.all(out[0, mask] == 0.0)


def test_identity_preexp_multiple_items() -> None:
    """IdentityPreMatrix handles batched item indices."""
    pre = IdentityPreMatrix(n_items=5, n_features=8)
    out = pre.apply(np.array([0, 2, 4]))
    assert out.shape == (3, 8)
    assert out[0, 0] == 1.0
    assert out[1, 2] == 1.0
    assert out[2, 4] == 1.0


def test_identity_preexp_rejects_out_of_range() -> None:
    """Out-of-range indices raise — no silent clipping (Constitution I)."""
    pre = IdentityPreMatrix(n_items=5, n_features=8)
    with pytest.raises(ValueError, match="out of range"):
        pre.apply(np.array([5]))


def test_identity_preexp_rejects_non_1d() -> None:
    pre = IdentityPreMatrix(n_items=5, n_features=8)
    with pytest.raises(ValueError, match="1-D"):
        pre.apply(np.array([[0, 1]]))


def test_identity_preexp_conforms_to_mfcprematrix_protocol() -> None:
    pre = IdentityPreMatrix(n_items=3, n_features=5)
    assert isinstance(pre, MFCPreMatrix)
    assert pre.n_items == 3
    assert pre.n_features == 5


def test_embedding_preexp_returns_supplied_embedding_row() -> None:
    """EmbeddingPreMatrix.apply reads rows from the supplied embeddings matrix."""
    embeddings = np.array([
        [1.0, 0.0, 0.0],  # item 0
        [0.5, 0.5, 0.0],  # item 1 — overlaps with item 0
        [0.0, 0.0, 1.0],  # item 2
    ])
    pre = EmbeddingPreMatrix(embeddings)
    out = pre.apply(np.array([0, 1]))
    np.testing.assert_array_almost_equal(out, embeddings[[0, 1]])


def test_embedding_preexp_rejects_out_of_range() -> None:
    embeddings = np.eye(3)
    pre = EmbeddingPreMatrix(embeddings)
    with pytest.raises(ValueError, match="out of range"):
        pre.apply(np.array([3]))


def test_embedding_preexp_from_parquet_raises_notimplemented() -> None:
    """USE integration is out of scope for feature 002."""
    with pytest.raises(NotImplementedError, match="Xu et al. 2026"):
        EmbeddingPreMatrix.from_parquet("anything.parquet")


def test_embedding_preexp_conforms_to_mfcprematrix_protocol() -> None:
    pre = EmbeddingPreMatrix(np.eye(4))
    assert isinstance(pre, MFCPreMatrix)
    assert pre.n_items == 4
    assert pre.n_features == 4


def test_embedding_preexp_output_differs_from_identity_when_embeddings_overlap() -> None:
    """A4-adjacent sanity check: embeddings that encode semantic overlap
    produce different trajectories than identity.
    """
    overlapping = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.8, 0.6, 0.0, 0.0],   # item 1 is similar to item 0
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    emb = EmbeddingPreMatrix(overlapping)
    idn = IdentityPreMatrix(n_items=4, n_features=4)

    indices = np.array([0, 1])
    emb_out = emb.apply(indices)
    idn_out = idn.apply(indices)

    # Items 0 and 1 under the embedding matrix have non-zero dot product,
    # whereas under the identity matrix they are orthogonal.
    emb_dot = float(np.dot(emb_out[0], emb_out[1]))
    idn_dot = float(np.dot(idn_out[0], idn_out[1]))
    assert emb_dot > 0.0
    assert idn_dot == 0.0
