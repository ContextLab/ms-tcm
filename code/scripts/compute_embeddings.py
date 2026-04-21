"""Compute sentence-transformer embeddings for every unique word in the FRFR
category dataset. Cached to ``data/processed/embeddings/frfr_category.parquet``.

Default embedder: ``sentence-transformers/all-MiniLM-L6-v2`` (no auth needed,
~90 MB; produces 384-dim embeddings). Swap with ``--model <hf-model-id>`` to
use a different backbone (e.g. ``google/embeddinggemma-300m``, which requires
a Hugging Face login + license acceptance).

The script is idempotent: if the output parquet already exists and contains
all of the dataset's unique words, it exits without re-running the model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model id.",
    )
    parser.add_argument(
        "--dataset", default="data/raw/frfr_category",
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--out", default="data/processed/embeddings/frfr_category.parquet",
        help="Output Parquet path.",
    )
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from ms_tcm import load_dataset
    ds = load_dataset(args.dataset)
    words = sorted(set(ds.presented.to_pandas()["word"].tolist()))
    print(f"{len(words)} unique words in {args.dataset}")

    if out_path.exists() and not args.force_rerun:
        existing = pd.read_parquet(out_path)
        if set(existing["word"].tolist()) >= set(words):
            print(f"Cache hit: {out_path} covers every word; use --force-rerun to recompute.")
            return 0
        print("Cache exists but does not cover all words; recomputing.")

    print(f"Loading model {args.model}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    print(f"Embedding {len(words)} words...")
    vecs = model.encode(words, show_progress_bar=False, convert_to_numpy=True)
    d = vecs.shape[1]
    print(f"d = {d}")

    # Write as (word, embedding_as_list) with a d-column marker in metadata.
    table = pa.Table.from_pandas(
        pd.DataFrame({
            "word": words,
            "embedding": [v.tolist() for v in vecs.astype(np.float32)],
        }),
        preserve_index=False,
    )
    pq.write_table(table, out_path, compression="zstd", compression_level=1)
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
