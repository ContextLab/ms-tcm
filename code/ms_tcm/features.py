"""Multi-hot feature encoder for MS-TCM (data-model.md section 2).

Pure, stateless encoder that maps a presented-words table to a fixed-width
float64 array. Total dimensionality d = 71:

    index 0                     -> reserved list-start slot (always zero in a
                                   presented word's features; the encoding
                                   loop uses e_start = one-hot at column 0 as
                                   the initial context vector)
    indices 1..16               -> category one-hot (15 FRFR categories + 1
                                   reserved UNKNOWN slot)
    indices 17..18              -> size one-hot ("small", "large")
    indices 19..44              -> first_letter one-hot A..Z
    indices 45..54              -> word_length one-hot over {3..12}
    indices 55..58              -> color_r bin (4 bins; -1 => all zero)
    indices 59..62              -> color_g bin
    indices 63..66              -> color_b bin
    indices 67..68              -> pos_x one-hot (left/right; NaN => zero)
    indices 69..70              -> pos_y one-hot (top/bottom; NaN => zero)

Changing any of the constants below is a MINOR schema-version bump per
contracts/dataset-schema.md section 6.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa

# Ordering is frozen: it determines the column-layout of every feature vector.
CATEGORY_BINS: tuple[str, ...] = (
    "BODY PARTS",
    "BUILDING RELATED",
    "CITIES",
    "CLOTHING",
    "COUNTRIES",
    "FLOWERS",
    "FRUITS",
    "INSECTS",
    "INSTRUMENTS",
    "KITCHEN-RELATED",
    "MAMMALS",
    "STATES",
    "TOOLS",
    "TREES",
    "VEGETABLES",
)

SIZE_BINS: tuple[str, ...] = ("small", "large")

LETTER_BINS: tuple[str, ...] = tuple(chr(ord("A") + i) for i in range(26))

LENGTH_BINS: tuple[int, ...] = tuple(range(3, 13))  # 3..12 inclusive (10 bins)

COLOR_BIN_WIDTH: int = 64  # 4 bins over [0, 255]
N_COLOR_BINS: int = 4

POSITION_BIN_CUT: float = 0.5
N_POSITION_BINS: int = 2

# Slot widths (must match the module docstring).
LIST_START_WIDTH = 1
CATEGORY_WIDTH = len(CATEGORY_BINS) + 1  # +1 for UNKNOWN slot
SIZE_WIDTH = len(SIZE_BINS)
LETTER_WIDTH = len(LETTER_BINS)
LENGTH_WIDTH = len(LENGTH_BINS)
COLOR_R_WIDTH = N_COLOR_BINS
COLOR_G_WIDTH = N_COLOR_BINS
COLOR_B_WIDTH = N_COLOR_BINS
POS_X_WIDTH = N_POSITION_BINS
POS_Y_WIDTH = N_POSITION_BINS

FEATURE_DIM: int = (
    LIST_START_WIDTH + CATEGORY_WIDTH + SIZE_WIDTH + LETTER_WIDTH + LENGTH_WIDTH
    + COLOR_R_WIDTH + COLOR_G_WIDTH + COLOR_B_WIDTH + POS_X_WIDTH + POS_Y_WIDTH
)
assert FEATURE_DIM == 71, f"FEATURE_DIM drifted: {FEATURE_DIM}"


# Precomputed slice offsets.
_OFFSETS: dict[str, tuple[int, int]] = {}
_cursor = 0
for name, width in (
    ("list_start", LIST_START_WIDTH),
    ("category", CATEGORY_WIDTH),
    ("size", SIZE_WIDTH),
    ("letter", LETTER_WIDTH),
    ("length", LENGTH_WIDTH),
    ("color_r", COLOR_R_WIDTH),
    ("color_g", COLOR_G_WIDTH),
    ("color_b", COLOR_B_WIDTH),
    ("pos_x", POS_X_WIDTH),
    ("pos_y", POS_Y_WIDTH),
):
    _OFFSETS[name] = (_cursor, _cursor + width)
    _cursor += width


def list_start_vector() -> np.ndarray:
    """The reserved list-start unit vector e_start (data-model.md section 4.1)."""
    v = np.zeros(FEATURE_DIM, dtype=np.float64)
    v[0] = 1.0
    return v


def _index_in(seq: tuple, value) -> int:
    try:
        return seq.index(value)
    except ValueError:
        return -1


def _color_bin(value: int) -> int:
    """Map an RGB channel value to a 4-bin index; -1 sentinel => -1 (all-zero)."""
    if value < 0:
        return -1
    if value >= 255:
        return N_COLOR_BINS - 1
    return int(value) // COLOR_BIN_WIDTH


def _position_bin(value: float) -> int:
    """Map a normalized display coordinate to {0, 1}; NaN => -1."""
    if value != value:  # NaN check
        return -1
    return 0 if value < POSITION_BIN_CUT else 1


def _encode_row(row: pd.Series) -> np.ndarray:
    v = np.zeros(FEATURE_DIM, dtype=np.float64)
    # index 0 reserved for list_start; always zero here.

    # Category
    lo, hi = _OFFSETS["category"]
    cat = row["category"]
    idx = _index_in(CATEGORY_BINS, cat)
    if idx < 0:
        # Known-unknown -> last slot (the +1 UNKNOWN).
        v[hi - 1] = 1.0
    else:
        v[lo + idx] = 1.0

    # Size
    lo, hi = _OFFSETS["size"]
    idx = _index_in(SIZE_BINS, row["size"])
    if idx >= 0:
        v[lo + idx] = 1.0

    # First letter
    lo, hi = _OFFSETS["letter"]
    letter = str(row["first_letter"]).upper()
    idx = _index_in(LETTER_BINS, letter)
    if idx >= 0:
        v[lo + idx] = 1.0

    # Word length
    lo, hi = _OFFSETS["length"]
    wl = int(row["word_length"])
    if wl not in LENGTH_BINS:
        raise ValueError(
            f"word_length={wl!r} out of supported range {LENGTH_BINS}"
        )
    v[lo + LENGTH_BINS.index(wl)] = 1.0

    # Color RGB
    for ch, field in (("color_r", "color_r"), ("color_g", "color_g"), ("color_b", "color_b")):
        lo, _ = _OFFSETS[ch]
        bin_idx = _color_bin(int(row[field]))
        if bin_idx >= 0:
            v[lo + bin_idx] = 1.0

    # Position x / y
    for ch, field in (("pos_x", "pos_x"), ("pos_y", "pos_y")):
        lo, _ = _OFFSETS[ch]
        bin_idx = _position_bin(float(row[field]))
        if bin_idx >= 0:
            v[lo + bin_idx] = 1.0

    return v


def _encode_features_iterrows(presented, *, normalize: bool = True) -> np.ndarray:
    """Legacy iterrows encoder — kept ONLY as a golden reference for the
    bit-identity test (test_features.py::test_vectorization_bit_identity).

    Not part of the public API. Do not call this from production code — the
    vectorized ``encode_features`` below is ~30x faster and produces
    bit-identical output (FR-030 Tier 1).
    """
    if isinstance(presented, pa.Table):
        df = presented.to_pandas()
    else:
        df = presented
    out = np.zeros((len(df), FEATURE_DIM), dtype=np.float64)
    for i, (_, row) in enumerate(df.iterrows()):
        out[i, :] = _encode_row(row)
    if normalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        out = out / norms
    return out


def encode_features(presented, *, normalize: bool = True) -> np.ndarray:
    """Return a (n_rows, FEATURE_DIM) float64 feature matrix (VECTORIZED).

    Accepts either a pyarrow Table or a pandas DataFrame with the columns
    defined in contracts/dataset-schema.md section 3. Vectorized
    implementation (FR-030 Tier 1): column-wise numpy scatter with no
    pandas.iterrows(), ~30x faster than the legacy row-by-row encoder. Bit-
    identical output (verified by test_features.py).

    By default each row is L2-normalized to unit length. This matters because
    TCM's drift equation c(t) = rho * c(t-1) + beta * c_in assumes unit-norm
    inputs so that rho = sqrt(1 - beta^2) preserves ||c(t)|| = 1 (see Howard
    & Kahana 2002, Eq. 3 and the surrounding discussion). Multi-hot raw
    features have norm sqrt(k) where k is the number of active slots; using
    those directly breaks the invariant and causes the similarity calculation
    to produce essentially-uniform softmax outputs regardless of beta. Pass
    ``normalize=False`` to skip normalization (e.g. in unit tests that
    supply their own unit-norm basis vectors).
    """
    if isinstance(presented, pa.Table):
        df = presented.to_pandas()
    else:
        df = presented
    n_rows = len(df)
    out = np.zeros((n_rows, FEATURE_DIM), dtype=np.float64)
    if n_rows == 0:
        return out

    row_idx = np.arange(n_rows, dtype=np.int64)

    # --- Category (index 1..16; UNKNOWN slot is the last one) ---
    lo, hi = _OFFSETS["category"]
    categories = df["category"].to_numpy()
    cat_to_idx = {c: i for i, c in enumerate(CATEGORY_BINS)}
    cat_cols = np.array(
        [cat_to_idx.get(c, CATEGORY_WIDTH - 1) for c in categories],
        dtype=np.int64,
    )
    out[row_idx, lo + cat_cols] = 1.0

    # --- Size (index 17..18) ---
    lo, _ = _OFFSETS["size"]
    sizes = df["size"].to_numpy()
    size_to_idx = {s: i for i, s in enumerate(SIZE_BINS)}
    for i, s in enumerate(sizes):
        idx = size_to_idx.get(s, -1)
        if idx >= 0:
            out[i, lo + idx] = 1.0

    # --- First letter (index 19..44) ---
    lo, _ = _OFFSETS["letter"]
    letters = df["first_letter"].astype(str).str.upper().to_numpy()
    letter_to_idx = {L: i for i, L in enumerate(LETTER_BINS)}
    for i, L in enumerate(letters):
        idx = letter_to_idx.get(L, -1)
        if idx >= 0:
            out[i, lo + idx] = 1.0

    # --- Word length (index 45..54) ---
    lo, _ = _OFFSETS["length"]
    lengths = df["word_length"].to_numpy().astype(np.int64)
    len_to_idx = {wl: i for i, wl in enumerate(LENGTH_BINS)}
    for i, wl in enumerate(lengths):
        if wl not in len_to_idx:
            raise ValueError(
                f"word_length={wl!r} out of supported range {LENGTH_BINS}"
            )
        out[i, lo + len_to_idx[wl]] = 1.0

    # --- Color RGB (three 4-bin blocks) ---
    for ch, field_name in (("color_r", "color_r"), ("color_g", "color_g"), ("color_b", "color_b")):
        lo, _ = _OFFSETS[ch]
        values = df[field_name].to_numpy().astype(np.int64)
        for i, val in enumerate(values):
            bin_idx = _color_bin(int(val))
            if bin_idx >= 0:
                out[i, lo + bin_idx] = 1.0

    # --- Position x / y (2-bin blocks) ---
    for ch, field_name in (("pos_x", "pos_x"), ("pos_y", "pos_y")):
        lo, _ = _OFFSETS[ch]
        values = df[field_name].to_numpy().astype(np.float64)
        for i, val in enumerate(values):
            bin_idx = _position_bin(float(val))
            if bin_idx >= 0:
                out[i, lo + bin_idx] = 1.0

    if normalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        out = out / norms
    return out


class DatasetFeatureCache:
    """Per-dataset cache of the feature matrix and M^FC_pre (FR-030 Tier 1).

    During a fit we evaluate ``dataset_log_likelihood`` hundreds to thousands
    of times (5 restarts x 1000 bootstrap draws x K likelihood evals). Each
    call runs ``encode_features`` on the same pyarrow-backed Dataset. We
    cache the feature matrix and the pre-experimental matrix keyed by the
    identity of the Dataset object so the encode step becomes O(1) after the
    first call. Invalidation happens automatically when a new Dataset is
    constructed (different ``id()``).

    Thread-safety: each worker process in the bootstrap Pool has its own
    cache (no shared state across processes). Within a process, the cache is
    not protected by a lock — current usage is single-threaded per worker.
    """

    def __init__(self) -> None:
        self._cache: dict[int, tuple[np.ndarray, object]] = {}

    def get_or_compute(self, dataset, pre_matrix) -> np.ndarray:
        """Return the cached feature matrix for ``dataset``; compute if new."""
        key = id(dataset)
        if key in self._cache:
            cached_features, cached_pre = self._cache[key]
            if cached_pre is pre_matrix:
                return cached_features
            # Pre-matrix changed; recompute (rare path — tests only).
        features = encode_features(dataset.presented)
        self._cache[key] = (features, pre_matrix)
        return features

    def clear(self) -> None:
        self._cache.clear()
