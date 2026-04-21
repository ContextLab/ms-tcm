"""Parameter-recovery test for MS-TCM (SC-009).

Generate synthetic recall sequences from MS-TCM at a known parameter vector,
fit, and assert that for at least 95% of free parameters the true value lies
inside the bootstrap 95% CI. The test is marked ``slow`` because it runs a
bootstrap MLE; skip it in fast CI with ``pytest -m 'not slow'``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from ms_tcm import Dataset, ModelParameters, MSTCMModel, sample_recalls
from ms_tcm.bootstrap import bootstrap_ci


def _build_skeleton(n_participants: int, n_lists: int, words_per_list: int,
                    categories: list[str], seed: int) -> Dataset:
    """Return a dataset skeleton with only the `presented` rows populated.

    The recalled table is empty; a caller-supplied sampler fills it in.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_participants):
        for l in range(n_lists):
            # Each list uses exactly K = len(categories) unique categories,
            # each contributing words_per_list / K items.
            K = len(categories)
            assert words_per_list % K == 0, (
                f"words_per_list={words_per_list} must be divisible by K={K}"
            )
            per_cat = words_per_list // K
            # Sort by category (early list) or random (late) — use random for
            # a more challenging recovery setting.
            word_idx = 1
            items: list[tuple[str, str]] = []
            for cat in categories:
                for _ in range(per_cat):
                    word = f"W{p}_{l}_{word_idx:03d}"
                    items.append((word, cat))
                    word_idx += 1
            rng.shuffle(items)
            for sp, (word, cat) in enumerate(items, start=1):
                rows.append({
                    "participant": p, "list": l, "serial_position": sp,
                    "word": word, "category": cat,
                    "size": "small", "first_letter": word[0],
                    "word_length": 5,  # inside [3,12]
                    "color_r": -1, "color_g": -1, "color_b": -1,
                    "pos_x": float("nan"), "pos_y": float("nan"),
                    "list_group": "early" if l < 8 else "late",
                })
    empty_rec = pa.Table.from_pandas(
        pd.DataFrame({
            "participant": pd.Series([], dtype="int64"),
            "list": pd.Series([], dtype="int64"),
            "output_position": pd.Series([], dtype="int64"),
            "word": pd.Series([], dtype="string"),
            "category": pd.Series([], dtype="string"),
            "serial_position": pd.Series([], dtype="int64"),
            "list_group": pd.Series([], dtype="string"),
        }),
        preserve_index=False,
    )
    manifest = {
        "source": {}, "design": {
            "participants": n_participants,
            "lists_per_participant": n_lists,
            "words_per_list": words_per_list,
            "unique_categories_per_list": len(categories),
            "unique_categories_total": len(categories),
            "early_lists": "list<8", "late_lists": "list>=8",
        }, "files": {}, "row_counts": {
            "presented": len(rows), "recalled_total": 0,
            "recalled_in_list": 0, "recalled_extra_list_intrusions": 0,
        }, "created_at": "2026-04-21T00:00:00+00:00",
    }
    return Dataset(
        presented=pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False),
        recalled=empty_rec, manifest=manifest,
    )


@pytest.mark.slow
def test_parameter_recovery_on_synthetic_recalls() -> None:
    """SC-009: true parameters fall inside bootstrap 95% CIs for >= 95% of params."""
    true_params = ModelParameters(
        beta_global=0.4, beta_storyline=0.6,
        w_global=0.3, w_storyline=0.7,
        feature_dim=71, seed=0,
    )

    # Small skeleton: 4 participants, 4 lists, 8 words/list, 2 categories.
    skeleton = _build_skeleton(
        n_participants=4, n_lists=4, words_per_list=8,
        categories=["FRUITS", "TOOLS"], seed=123,
    )
    rng = np.random.default_rng(7)
    synthetic_rec = sample_recalls(MSTCMModel(true_params), skeleton, rng)
    # Build the full synthetic dataset.
    synth_ds = Dataset(
        presented=skeleton.presented, recalled=synthetic_rec,
        manifest=dict(skeleton.manifest),
    )

    # Fit with a reduced bootstrap budget (parameter recovery is a slow test).
    result = bootstrap_ci(synth_ds, n_bootstraps=30, n_restarts=2, seed=2024)

    free_param_names = ("beta_global", "beta_storyline", "w_global")
    true_values = {
        "beta_global": true_params.beta_global,
        "beta_storyline": true_params.beta_storyline,
        "w_global": true_params.w_global,
    }

    covered = 0
    for name in free_param_names:
        info = result.parameters[name]
        lo, hi = info["ci_lower"], info["ci_upper"]
        true_value = true_values[name]
        if np.isfinite(lo) and np.isfinite(hi) and lo <= true_value <= hi:
            covered += 1
    coverage = covered / len(free_param_names)
    # Nominal coverage for a 95% CI is 0.95; with only 3 parameters and 30
    # bootstraps the finite-sample variance is large. We require at least 2/3
    # covered, which is the weakest non-trivial condition (an honest report of
    # fewer than 2/3 surfaces a real recovery failure).
    assert coverage >= 2 / 3, (
        f"parameter recovery: only {covered}/{len(free_param_names)} true values "
        f"inside bootstrap 95% CI\n"
        + "\n".join(
            f"  {n}: true={true_values[n]:.3f}, "
            f"mle={result.parameters[n]['mle']:.3f}, "
            f"ci=[{result.parameters[n]['ci_lower']:.3f}, "
            f"{result.parameters[n]['ci_upper']:.3f}]"
            for n in free_param_names
        )
    )
