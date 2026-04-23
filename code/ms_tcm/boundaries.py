"""Boundary-driven state transitions for v6 hierarchical CMR.

v6 §2.2-2.4:

- Event boundary within a storyline (Eq 4): c^item <- c^story
- Storyline switch (Eq 5): cache outgoing c^story to M^SC, c^item <- c^story_new
- Storyline return (Eq 6): c^story <- lambda * c_tilde^story + (1 - lambda) * c^story_prev
  followed by Eq 7: c^item <- c^story

The caller (orchestrator in ``ms_tcm.hcmr``) determines which transition
applies based on (e(i), s(i)) vs. (e(i-1), s(i-1)) and the set of storylines
seen so far in the current list.
"""

from __future__ import annotations

import numpy as np

from ms_tcm.matrices import read_cached_story, sc_update


def apply_event_boundary(c_item: np.ndarray, c_story: np.ndarray) -> np.ndarray:
    """v6 Eq 4: at an event boundary within a storyline, c^item <- c^story.

    Returns a copy (callers may mutate the result independently of c_story).
    """
    c_item = np.asarray(c_item, dtype=np.float64)
    c_story = np.asarray(c_story, dtype=np.float64)
    if c_item.shape != c_story.shape:
        raise ValueError(
            f"apply_event_boundary: shape mismatch c_item={c_item.shape} vs c_story={c_story.shape}"
        )
    return c_story.copy()


def apply_storyline_switch(
    m_sc: np.ndarray,
    outgoing_idx: int,
    c_story_out: np.ndarray,
    c_story_new: np.ndarray,
    *,
    n_storylines: int,
) -> tuple[np.ndarray, np.ndarray]:
    """v6 Eq 5: cache outgoing storyline context to M^SC; sync c^item to new storyline.

    Returns (c_item_new, m_sc_updated). ``m_sc`` is mutated in place AND
    returned for caller convenience (so orchestrator code can use either
    pattern idiomatically).
    """
    if not (0 <= outgoing_idx < n_storylines):
        raise ValueError(
            f"apply_storyline_switch: outgoing_idx={outgoing_idx} out of range "
            f"[0, {n_storylines})"
        )
    onehot = np.zeros(n_storylines, dtype=np.float64)
    onehot[outgoing_idx] = 1.0
    # Cache outgoing storyline to M^SC.
    sc_update(m_sc, onehot, c_story_out)
    # Synchronize item-level context to the new storyline's context.
    c_item_new = np.asarray(c_story_new, dtype=np.float64).copy()
    return c_item_new, m_sc


def apply_storyline_return(
    m_sc: np.ndarray,
    returning_idx: int,
    c_story_prev: np.ndarray,
    lambda_reinstate: float,
    *,
    n_storylines: int,
) -> np.ndarray:
    """v6 Eq 6: c^story_new = lambda * c_tilde^story_s(i) + (1 - lambda) * c^story_prev.

    ``c_tilde^story_s(i)`` is the cached context for the returning storyline,
    read from M^SC. Returns c^story_new; the caller is responsible for
    subsequently applying Eq 7 (c^item <- c^story) via ``apply_event_boundary``.

    When lambda_reinstate = 0, this is a no-op (returns c_story_prev copy).
    When lambda_reinstate = 1, this fully restores the cached context.
    When the storyline has never been cached (never switched out), the
    cached vector is zero and c^story_new = (1 - lambda) * c_story_prev.
    """
    if not (0.0 <= lambda_reinstate <= 1.0):
        raise ValueError(
            f"lambda_reinstate must be in [0, 1]; got {lambda_reinstate!r}"
        )
    if not (0 <= returning_idx < n_storylines):
        raise ValueError(
            f"apply_storyline_return: returning_idx={returning_idx} out of range "
            f"[0, {n_storylines})"
        )

    onehot = np.zeros(n_storylines, dtype=np.float64)
    onehot[returning_idx] = 1.0
    c_cached = read_cached_story(m_sc, onehot)

    c_story_prev = np.asarray(c_story_prev, dtype=np.float64)
    return lambda_reinstate * c_cached + (1.0 - lambda_reinstate) * c_story_prev
