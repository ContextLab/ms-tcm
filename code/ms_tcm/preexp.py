"""Pre-experimental matrix M^FC_pre.

v6 §1.5 introduces M^FC_pre as a configurable object that maps item one-hot
vectors to pre-experimental context. Two concrete choices:

- Option 1 — identity matrix: each item's pre-experimental context is
  orthogonal (no semantic overlap). Used for the FRFR-category worked
  example and the C&Z 2025 free-recall baseline.
- Option 3 — pre-trained embeddings: each item's pre-experimental context
  is its embedding (e.g. USE sentence embeddings for naturalistic stimuli).
  The Xu et al. 2026 cued-recall fit would use this path; the actual USE
  loader is stubbed in this feature per spec Non-goals.

Spec: specs/002-ms-tcm-v6-hcmr/spec.md FR-007, User Story 4.
Data model: specs/002-ms-tcm-v6-hcmr/data-model.md §2.
Contract: specs/002-ms-tcm-v6-hcmr/contracts/model-api.md §2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class MFCPreMatrix(Protocol):
    """Abstract contract for a pre-experimental item-to-context matrix (v6 §1.5)."""

    n_features: int  # feature-vector dimensionality d
    n_items: int     # number of distinct item identities this matrix covers

    def apply(self, item_indices: np.ndarray) -> np.ndarray:
        """Return M^FC_pre @ f_i for each item index, shape (len(indices), n_features)."""
        ...


class IdentityPreMatrix:
    """v6 §1.5 Option 1: each item's pre-experimental context is its own one-hot.

    No semantic overlap between items. Used in the FRFR-category worked
    example and in any run where the user wants to isolate the λ mechanism
    from pre-experimental semantic overlap (cf. v6 §6.1 identity-vs-embedding
    diagnostic).
    """

    def __init__(self, n_items: int, n_features: int) -> None:
        if n_items <= 0:
            raise ValueError(f"n_items must be positive; got {n_items!r}")
        if n_features <= 0:
            raise ValueError(f"n_features must be positive; got {n_features!r}")
        self.n_items = int(n_items)
        self.n_features = int(n_features)

    def apply(self, item_indices: np.ndarray) -> np.ndarray:
        """Return a (len(indices), n_features) matrix of one-hot rows.

        Each row has a single 1.0 at column ``item_indices[i]`` modulo
        ``n_features`` (if n_items > n_features the surplus item identities
        simply wrap; typically n_items == n_features).
        """
        item_indices = np.asarray(item_indices, dtype=np.int64)
        if item_indices.ndim != 1:
            raise ValueError(
                f"item_indices must be 1-D; got shape {item_indices.shape}"
            )
        out = np.zeros((item_indices.shape[0], self.n_features), dtype=np.float64)
        # Guard against out-of-range indices; v6 needs an explicit error
        # rather than silent clipping (Constitution I).
        if np.any(item_indices < 0) or np.any(item_indices >= self.n_items):
            raise ValueError(
                f"item_indices out of range for IdentityPreMatrix(n_items="
                f"{self.n_items}); got min={int(item_indices.min())}, "
                f"max={int(item_indices.max())}"
            )
        # Map to feature columns. For n_items == n_features this is the
        # identity. For n_items < n_features we place the one-hot at column
        # = item index (leaving the remaining feature dimensions at 0).
        target_cols = item_indices % self.n_features
        out[np.arange(item_indices.shape[0]), target_cols] = 1.0
        return out


class EmbeddingPreMatrix:
    """v6 §1.5 Option 3: each item's pre-experimental context is a row of a
    pre-computed embeddings matrix (e.g. USE sentence embeddings for the
    Xu et al. 2026 naturalistic cued-recall case).

    This class provides the *interface* for plugging embeddings into the
    model; actual USE loading and sentence-embedding computation are OUT OF
    SCOPE for feature 002 (spec §Non-goals). A caller may supply an
    arbitrary precomputed embeddings matrix at construction time.
    """

    def __init__(
        self, embeddings: np.ndarray, item_to_index: dict[str, int] | None = None,
    ) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D (n_items, n_features); "
                f"got shape {embeddings.shape}"
            )
        self.embeddings = embeddings
        self.n_items, self.n_features = embeddings.shape
        self.item_to_index = item_to_index or {}

    def apply(self, item_indices: np.ndarray) -> np.ndarray:
        item_indices = np.asarray(item_indices, dtype=np.int64)
        if item_indices.ndim != 1:
            raise ValueError(
                f"item_indices must be 1-D; got shape {item_indices.shape}"
            )
        if np.any(item_indices < 0) or np.any(item_indices >= self.n_items):
            raise ValueError(
                f"item_indices out of range for EmbeddingPreMatrix(n_items="
                f"{self.n_items}); got min={int(item_indices.min())}, "
                f"max={int(item_indices.max())}"
            )
        return self.embeddings[item_indices].copy()

    @classmethod
    def from_parquet(cls, path: str | Path) -> "EmbeddingPreMatrix":
        """Load an embeddings matrix from a Parquet file.

        NOT IMPLEMENTED in feature 002 per spec §Non-goals. The actual USE
        integration lands alongside the Xu et al. 2026 cued-recall dataset.
        The stub raises NotImplementedError so callers fail fast rather than
        silently mis-loading.
        """
        raise NotImplementedError(
            f"EmbeddingPreMatrix.from_parquet(path={path!r}) is reserved for "
            "the Xu et al. 2026 cued-recall feature and is not implemented in "
            "002-ms-tcm-v6-hcmr (see spec §Non-goals). Construct an "
            "EmbeddingPreMatrix directly with a precomputed numpy array instead."
        )
