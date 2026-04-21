"""Dataset dataclass, load_dataset, save_dataset.

See contracts/model-api.md section 3 and contracts/dataset-schema.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from ms_tcm.io import read_json, sha256_file, write_json_sorted


@dataclass(frozen=True)
class Dataset:
    presented: pa.Table
    recalled: pa.Table
    manifest: dict[str, Any]

    @property
    def num_participants(self) -> int:
        return int(self.manifest["design"]["participants"])

    @property
    def num_lists_per_participant(self) -> int:
        return int(self.manifest["design"]["lists_per_participant"])

    @property
    def num_words_per_list(self) -> int:
        return int(self.manifest["design"]["words_per_list"])

    @property
    def feature_dim(self) -> int:
        from ms_tcm.features import FEATURE_DIM
        return FEATURE_DIM

    def categories_in_list(self, participant: int, list_: int) -> list[str]:
        df = self.presented.to_pandas()
        rows = df[(df["participant"] == participant) & (df["list"] == list_)]
        return sorted(set(rows["category"].tolist()))

    def iter_lists(self) -> Iterator[tuple[int, int, pa.Table, pa.Table]]:
        """Yield (participant, list_, presented_subtable, recalled_subtable)."""
        pdf = self.presented.to_pandas()
        rdf = self.recalled.to_pandas()
        keys = sorted(
            set(zip(pdf["participant"].tolist(), pdf["list"].tolist()))
        )
        for p, l in keys:
            pres_sub = pdf[(pdf["participant"] == p) & (pdf["list"] == l)]
            rec_sub = rdf[(rdf["participant"] == p) & (rdf["list"] == l)]
            yield p, l, pa.Table.from_pandas(pres_sub, preserve_index=False), \
                  pa.Table.from_pandas(rec_sub, preserve_index=False)


_PARQUET_OPTIONS: dict = dict(
    compression="zstd",
    compression_level=1,
    use_dictionary=False,
    write_statistics=False,
    version="2.6",
)


def _write_parquet_deterministic(table: pa.Table, path: Path) -> None:
    pq.write_table(table, path, **_PARQUET_OPTIONS)


def load_dataset(path: str | os.PathLike[str]) -> Dataset:
    """Read a dataset directory. Verifies manifest SHA-256 hashes."""
    p = Path(path)
    manifest = read_json(p / "manifest.json")
    # Verify hashes for every listed file that exists.
    for fname, entry in (manifest.get("files") or {}).items():
        if not isinstance(entry, dict) or "sha256" not in entry:
            continue
        fpath = p / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"manifest lists {fname!r} but file is missing at {fpath}"
            )
        actual = sha256_file(fpath)
        if actual != entry["sha256"]:
            raise ValueError(
                f"manifest hash mismatch for {fname!r}: "
                f"manifest={entry['sha256']!r}, disk={actual!r}"
            )
    presented = pq.read_table(p / "presented.parquet")
    recalled = pq.read_table(p / "recalled.parquet")
    return Dataset(presented=presented, recalled=recalled, manifest=manifest)


def save_dataset(dataset: Dataset, path: str | os.PathLike[str]) -> None:
    """Write a dataset directory with deterministic Parquet + CSV + manifest.

    Updates the manifest's ``files`` block with fresh SHA-256 hashes and byte
    sizes computed from the bytes written in this call.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    # Sort presented/recalled before writing (primary keys; deterministic output).
    pres_sorted = dataset.presented.sort_by(
        [("participant", "ascending"), ("list", "ascending"), ("serial_position", "ascending")]
    )
    rec_sorted = dataset.recalled.sort_by(
        [("participant", "ascending"), ("list", "ascending"), ("output_position", "ascending")]
    )

    _write_parquet_deterministic(pres_sorted, p / "presented.parquet")
    _write_parquet_deterministic(rec_sorted, p / "recalled.parquet")
    pres_sorted.to_pandas().to_csv(p / "presented.csv", index=False)
    rec_sorted.to_pandas().to_csv(p / "recalled.csv", index=False)

    # Recompute hashes.
    files_block: dict[str, dict] = {}
    for fname in ("presented.parquet", "recalled.parquet", "presented.csv", "recalled.csv"):
        fpath = p / fname
        files_block[fname] = {
            "sha256": sha256_file(fpath),
            "size_bytes": fpath.stat().st_size,
        }

    manifest = dict(dataset.manifest)
    manifest["files"] = files_block
    write_json_sorted(manifest, p / "manifest.json")
