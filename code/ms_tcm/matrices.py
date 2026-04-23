"""Associative matrices M^IC, M^SC, M^FC_exp for v6 hierarchical CMR.

All updates operate in-place on preallocated arrays (Tier 1 performance:
avoid allocator churn during the encoding loop; contracts/fitter.md §6).

Canonical source: ``notes/two_level_cmr_v6.pdf`` Eqs 3, 5 and ``notes/CornZhan25.pdf``
Eq 2a for M^FC_exp.

Data model: ``specs/002-ms-tcm-v6-hcmr/data-model.md`` §5.
"""

from __future__ import annotations

import numpy as np


def ic_update(m_ic: np.ndarray, c_item: np.ndarray, f_i: np.ndarray) -> None:
    """In-place update M^IC += c_item · f_i^T (v6 Eq 3 / CMR Eq 2a).

    Shape convention: ``m_ic`` is (d, n_items); ``c_item`` is (d,); ``f_i``
    is (n_items,), typically a one-hot vector identifying the item.

    Effect: item i gets a new column contribution equal to the current
    item-level context ``c_item``.
    """
    c_item = np.asarray(c_item, dtype=np.float64)
    f_i = np.asarray(f_i, dtype=np.float64)
    if m_ic.shape != (c_item.shape[0], f_i.shape[0]):
        raise ValueError(
            f"ic_update: expected m_ic shape {(c_item.shape[0], f_i.shape[0])}, "
            f"got {m_ic.shape}"
        )
    # In-place outer product accumulation: np.outer returns a new array;
    # we add it into m_ic with ufunc += to keep the allocation bounded.
    m_ic += np.outer(c_item, f_i)


def sc_update(m_sc: np.ndarray, storyline_onehot: np.ndarray, c_story_out: np.ndarray) -> None:
    """In-place update ΔM^SC = g_s · c_story_out^T (v6 Eq 5).

    ``m_sc`` is (n_storylines, d); ``storyline_onehot`` is (n_storylines,)
    with a single 1.0 at the outgoing storyline's index; ``c_story_out`` is
    (d,). The caller applies this at storyline switches only.
    """
    storyline_onehot = np.asarray(storyline_onehot, dtype=np.float64)
    c_story_out = np.asarray(c_story_out, dtype=np.float64)
    if m_sc.shape != (storyline_onehot.shape[0], c_story_out.shape[0]):
        raise ValueError(
            f"sc_update: expected m_sc shape "
            f"{(storyline_onehot.shape[0], c_story_out.shape[0])}, got {m_sc.shape}"
        )
    m_sc += np.outer(storyline_onehot, c_story_out)


def fc_exp_update(m_fc_exp: np.ndarray, c_in: np.ndarray, f_i: np.ndarray) -> None:
    """In-place update ΔM^FC_exp = c_in · f_i^T (CMR Eq 2a, experimental branch).

    Shape: ``m_fc_exp`` is (d, n_items). Used to accumulate the experimental
    context-of-item matrix over encoding; the γ_fc mixture (v6 Eq 1.5.1) reads
    from this at encoding time.
    """
    c_in = np.asarray(c_in, dtype=np.float64)
    f_i = np.asarray(f_i, dtype=np.float64)
    if m_fc_exp.shape != (c_in.shape[0], f_i.shape[0]):
        raise ValueError(
            f"fc_exp_update: expected m_fc_exp shape "
            f"{(c_in.shape[0], f_i.shape[0])}, got {m_fc_exp.shape}"
        )
    m_fc_exp += np.outer(c_in, f_i)


def read_cached_story(m_sc: np.ndarray, storyline_onehot: np.ndarray) -> np.ndarray:
    """Read back the cached storyline context for the returning storyline.

    Implements the ``tilde c^story_s(i)`` term in v6 §2.4 Eq 6. Given M^SC
    of shape (n_storylines, d) and a one-hot ``storyline_onehot`` picking
    out the returning storyline, returns the cached d-vector.

    Mathematically this is ``M^SC.T @ storyline_onehot`` for the case of a
    fresh-per-return read; if the storyline has switched out multiple times
    (accumulation across returns), the read sums all prior caches. The
    unit-norm invariant on c_story makes repeated outer-product accumulations
    well-defined; the caller is responsible for ensuring unit-norm inputs.
    """
    storyline_onehot = np.asarray(storyline_onehot, dtype=np.float64)
    if m_sc.shape[0] != storyline_onehot.shape[0]:
        raise ValueError(
            f"read_cached_story: m_sc has {m_sc.shape[0]} storylines but "
            f"storyline_onehot has length {storyline_onehot.shape[0]}"
        )
    return m_sc.T @ storyline_onehot
