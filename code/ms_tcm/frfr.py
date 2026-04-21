"""Convenience loader for the bundled FRFR-category dataset.

See contracts/model-api.md section 5.
"""

from __future__ import annotations

import os
from pathlib import Path

from ms_tcm.dataset import Dataset, load_dataset

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "frfr_category"


def load_frfr_category(
    path: str | os.PathLike[str] = _DEFAULT_PATH,
) -> Dataset:
    """Load the Manning et al. (2023) category-condition dataset.

    Verifies that the loaded shape matches the paper's design: 30 participants,
    16 lists per participant, 16 words per list, 4 unique categories per list.
    Raises ``ValueError`` if any expectation is violated, so a reviewer cannot
    silently run the pipeline on a differently-shaped dataset.
    """
    ds = load_dataset(path)
    design = ds.manifest.get("design", {})
    expected = {
        "participants": 30,
        "lists_per_participant": 16,
        "words_per_list": 16,
        "unique_categories_per_list": 4,
    }
    for key, expected_value in expected.items():
        actual = design.get(key)
        if actual != expected_value:
            raise ValueError(
                f"FRFR-category dataset has unexpected {key}: "
                f"expected {expected_value!r}, got {actual!r}"
            )
    return ds
