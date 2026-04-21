"""Load precomputed word embeddings as a dict {word: np.ndarray}."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_embeddings(
    path: str = "data/processed/embeddings/frfr_category.parquet",
) -> dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {p}; run "
            f"`python code/scripts/compute_embeddings.py` first."
        )
    df = pd.read_parquet(p)
    return {
        str(row["word"]): np.asarray(row["embedding"], dtype=np.float64)
        for _, row in df.iterrows()
    }
