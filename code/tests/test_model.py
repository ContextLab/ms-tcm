"""End-to-end model tests: K=1 reduction, list-boundary reset, standard-TCM switch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ms_tcm import Dataset, ModelParameters, MSTCMModel


def _minimal_dataset(categories_per_list: list[list[str]]) -> Dataset:
    """Build a tiny in-memory Dataset for exercising the encoder without touching disk.

    ``categories_per_list[p][l]`` is the category-ordered word sequence for
    participant ``p``, list ``l``. Each category string is the word itself
    (one word per category for simplicity).
    """
    presented_rows = []
    recalled_rows = []
    for p, lists in enumerate([categories_per_list]):
        for l_idx, cats in enumerate(lists):
            for sp, cat in enumerate(cats, start=1):
                presented_rows.append({
                    "participant": p,
                    "list": l_idx,
                    "serial_position": sp,
                    "word": cat,
                    "category": cat,
                    "size": "small",
                    "first_letter": cat[0],
                    "word_length": len(cat),
                    "color_r": -1, "color_g": -1, "color_b": -1,
                    "pos_x": float("nan"), "pos_y": float("nan"),
                    "list_group": "early" if l_idx < 8 else "late",
                })
    manifest = {
        "source": {},
        "design": {
            "participants": 1,
            "lists_per_participant": len(categories_per_list),
            "words_per_list": len(categories_per_list[0]),
            "unique_categories_per_list": len(set(categories_per_list[0])),
            "unique_categories_total": len({w for lst in categories_per_list for w in lst}),
            "early_lists": "list<8",
            "late_lists": "list>=8",
        },
        "files": {},
        "row_counts": {
            "presented": len(presented_rows),
            "recalled_total": 0,
            "recalled_in_list": 0,
            "recalled_extra_list_intrusions": 0,
        },
        "created_at": "2026-04-21T00:00:00+00:00",
    }
    import pyarrow as pa
    presented_tbl = pa.Table.from_pandas(pd.DataFrame(presented_rows), preserve_index=False)
    recalled_tbl = pa.Table.from_pandas(
        pd.DataFrame(columns=[
            "participant", "list", "output_position", "word",
            "category", "serial_position", "list_group",
        ]).astype({
            "participant": "int64", "list": "int64", "output_position": "int64",
            "word": "string", "category": "string",
            "serial_position": "int64", "list_group": "string",
        }),
        preserve_index=False,
    )
    return Dataset(presented=presented_tbl, recalled=recalled_tbl, manifest=manifest)


def test_single_storyline_matches_standard_tcm() -> None:
    """K = 1 (one category) reduction: MS-TCM equals standard TCM within 1e-12.

    When every word belongs to the same category the storyline stream and the
    global stream receive the same inputs at the same times, so the composite
    context under MS-TCM equals w_G · c_G + w_S · c_S = c_G (because c_S == c_G
    when the same updates are applied to zero-initialized vectors with the
    same β), and standard TCM's composite is just c_G. Both must agree.
    """
    ds = _minimal_dataset([["FRUITS"] * 4])
    p_ms = ModelParameters(beta_global=0.5, beta_storyline=0.5, w_global=0.2, w_storyline=0.8)
    p_tcm = ModelParameters.standard_tcm(beta_global=0.5)

    state_ms = MSTCMModel(p_ms).encode(ds)
    state_tcm = MSTCMModel(p_tcm).encode(ds)

    # Composite trajectories should match within 1e-12.
    for key in state_ms.c_composite:
        np.testing.assert_allclose(
            state_ms.c_composite[key], state_tcm.c_composite[key], atol=1e-12,
        )


def test_list_boundary_resets_contexts() -> None:
    """At the first step of every list, c_G and every c_S equal the reserved
    list-start unit vector e_start = [1, 0, 0, ...] (data-model.md section 4.1).

    Initialization to e_start (rather than the zero vector) is what preserves
    the TCM unit-norm invariant and makes section-4.4's analytical anchor hold.
    Absence of cross-list leakage is demonstrated by the identity c_*(0) == e_start
    at every (participant, list) boundary.
    """
    from ms_tcm.features import list_start_vector
    e_start = list_start_vector()
    ds = _minimal_dataset([["CAT", "DOG"], ["BAR", "BOX"]])  # 2 lists, 2 words each
    state = MSTCMModel(ModelParameters()).encode(ds)

    for (p, l_), arr in state.c_global.items():
        np.testing.assert_allclose(
            arr[0], e_start, atol=1e-12,
            err_msg=f"c_G(0) != e_start for (p={p}, l={l_})",
        )
    for (p, l_), sl_dict in state.c_storyline.items():
        for cat, arr in sl_dict.items():
            np.testing.assert_allclose(
                arr[0], e_start, atol=1e-12,
                err_msg=f"c_S(0) != e_start for (p={p}, l={l_}, cat={cat})",
            )


def test_standard_tcm_mode_disables_storyline_contribution() -> None:
    """FR-006: standard_tcm() sets w_S = 0; composite = c_G exactly."""
    ds = _minimal_dataset([["CAT", "DOG", "BAR", "BOX"]])
    p_tcm = ModelParameters.standard_tcm(beta_global=0.5)
    state = MSTCMModel(p_tcm).encode(ds)

    # Under w_S = 0, composite must equal c_G[1:] (skip the pre-list zero).
    for key in state.c_composite:
        c_comp = state.c_composite[key]      # shape (W, d)
        c_g = state.c_global[key][1:]        # shape (W, d)
        np.testing.assert_allclose(c_comp, c_g, atol=1e-12)
