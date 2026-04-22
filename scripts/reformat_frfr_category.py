"""Reformat FRFR Experiment 2 (category condition) into the canonical MS-TCM format.

Source: https://github.com/ContextLab/FRFR-analyses (Manning et al., 2023)
Egg URL: https://www.dropbox.com/s/kliq92lta7mvqcc/exp2.egg?dl=1

This script downloads the raw .egg (quail HDF5 format), extracts only the fields
MS-TCM needs, and writes them to ``data/raw/frfr_category/`` as both Parquet
(canonical) and CSV (diff-able). A ``manifest.json`` records SHA-256 hashes of
the source egg and every output file so bit-exact reproducibility can be
verified without re-downloading.

Run once locally; the outputs are committed to the repository so downstream
analyses do not require network access.

Security note: this script uses ``pickle.loads`` solely to decode HDF5
attribute blobs embedded in the upstream .egg file, which was produced by
quail on Python 2 and stored `newstr` scalars as pickled bytes. The decode is
scoped to a fixed set of string-valued HDF5 attributes per list-item (word,
category, size, first letter); it does not load any code path from the
attributes. There is no safe alternative for reading these legacy blobs.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pickle  # noqa: S403 - scoped to HDF5 attribute decoding, see module docstring
import re
import urllib.request
from pathlib import Path

import h5py
import pandas as pd

EGG_URL = "https://www.dropbox.com/s/kliq92lta7mvqcc/exp2.egg?dl=1"


def _decode(v: object) -> object:
    """Decode a pytables/pickle-wrapped attribute value back to a Python scalar."""
    if isinstance(v, bytes) and b"ccopy_reg" in v:
        try:
            return str(pickle.loads(v))  # noqa: S301 - legacy HDF5 attr, see module docstring
        except Exception:
            m = re.search(rb"V([^\n]*)\n", v)
            return m.group(1).decode("utf-8") if m else None
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(egg_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    presented: list[dict] = []
    recalled: list[dict] = []
    with h5py.File(egg_path, "r") as f:
        for p in range(len(f["pres"].keys())):
            for lst in range(len(f[f"pres/i{p}"].keys())):
                g = f[f"pres/i{p}/i{lst}"]
                for w in sorted(g.keys(), key=lambda s: int(s[1:])):
                    a = dict(g[w].attrs)
                    color_a = dict(f[f"pres/i{p}/i{lst}/{w}/color"].attrs)
                    loc_a = dict(f[f"pres/i{p}/i{lst}/{w}/location"].attrs)
                    presented.append(
                        {
                            "participant": p,
                            "list": lst,
                            "serial_position": int(w[1:]) + 1,
                            "word": _decode(a["item"]),
                            "category": _decode(a["category"]),
                            "size": _decode(a["size"]),
                            "first_letter": _decode(a["firstLetter"]),
                            "word_length": int(a["wordLength"]),
                            "color_r": int(color_a.get("i0", -1)),
                            "color_g": int(color_a.get("i1", -1)),
                            "color_b": int(color_a.get("i2", -1)),
                            "pos_x": float(loc_a.get("i0", float("nan"))),
                            "pos_y": float(loc_a.get("i1", float("nan"))),
                        }
                    )
                rg = f[f"rec/i{p}/i{lst}"]
                out_idx = 0
                for r in sorted(rg.keys(), key=lambda s: int(s[1:])):
                    a = dict(rg[r].attrs)
                    item = _decode(a.get("item", b""))
                    if item is None or (isinstance(item, float) and item != item):
                        continue
                    tmp = a.get("temporal", None)
                    # Upstream convention (verified empirically by decoding the
                    # exp2.egg and cross-referencing recalled items against the
                    # presented table word-by-word): ``temporal`` is 0-indexed
                    # against the item index ``i{k}`` in the presented group,
                    # so temporal == 0 is the FIRST word of the list, and
                    # temporal == 15 is the LAST word of a 16-word list. We
                    # convert to our 1-indexed serial_position with +1. Values
                    # outside [0, 15] are extra-list intrusions (serial_position
                    # = 0 in our canonical format).
                    if tmp is None:
                        sp = 0
                    else:
                        try:
                            t = int(tmp)
                        except (TypeError, ValueError):
                            t = -1
                        sp = t + 1 if 0 <= t <= 15 else 0
                    recalled.append(
                        {
                            "participant": p,
                            "list": lst,
                            "output_position": out_idx + 1,
                            "word": item,
                            "category": _decode(a.get("category", b"")) or "",
                            "serial_position": sp,
                        }
                    )
                    out_idx += 1

    pres_df = (
        pd.DataFrame(presented)
        .sort_values(["participant", "list", "serial_position"])
        .reset_index(drop=True)
    )
    rec_df = (
        pd.DataFrame(recalled)
        .sort_values(["participant", "list", "output_position"])
        .reset_index(drop=True)
    )
    pres_df["list_group"] = (pres_df["list"] >= 8).map({False: "early", True: "late"})
    rec_df["list_group"] = (rec_df["list"] >= 8).map({False: "early", True: "late"})
    return pres_df, rec_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--egg", type=Path, default=None,
        help="Local path to exp2.egg; downloaded to a cache path under --out if omitted.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw" / "frfr_category",
        help="Output directory (default: data/raw/frfr_category).",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    egg_path = args.egg
    if egg_path is None:
        egg_path = args.out / "exp2.egg.cache"
        if not egg_path.exists():
            print(f"Downloading {EGG_URL} -> {egg_path}")
            urllib.request.urlretrieve(EGG_URL, egg_path)

    pres_df, rec_df = extract(egg_path)

    pres_df.to_parquet(args.out / "presented.parquet", compression="zstd", compression_level=1, index=False)
    rec_df.to_parquet(args.out / "recalled.parquet", compression="zstd", compression_level=1, index=False)
    pres_df.to_csv(args.out / "presented.csv", index=False)
    rec_df.to_csv(args.out / "recalled.csv", index=False)

    manifest = {
        "source": {
            "paper": "Manning et al. 2023 (https://github.com/ContextLab/FRFR-analyses)",
            "condition": "category (exp2)",
            "egg_url": EGG_URL,
            "egg_sha256": _sha256(egg_path),
        },
        "design": {
            "participants": int(pres_df["participant"].nunique()),
            "lists_per_participant": int(pres_df.groupby("participant")["list"].nunique().iloc[0]),
            "words_per_list": int(pres_df.groupby(["participant", "list"]).size().iloc[0]),
            "unique_categories_per_list": 4,
            "unique_categories_total": int(pres_df["category"].nunique()),
            "early_lists": "list < 8 (sorted by category)",
            "late_lists": "list >= 8 (random order)",
        },
        "files": {
            name: {"sha256": _sha256(args.out / name), "size_bytes": (args.out / name).stat().st_size}
            for name in ("presented.parquet", "recalled.parquet", "presented.csv", "recalled.csv")
        },
        "row_counts": {
            "presented": int(len(pres_df)),
            "recalled_total": int(len(rec_df)),
            "recalled_in_list": int((rec_df["serial_position"] > 0).sum()),
            "recalled_extra_list_intrusions": int((rec_df["serial_position"] == 0).sum()),
        },
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(args.out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"Wrote {args.out}")
    for name in ("presented.parquet", "recalled.parquet", "presented.csv", "recalled.csv", "manifest.json"):
        print(f"  {name}: {(args.out / name).stat().st_size:>8} bytes")


if __name__ == "__main__":
    main()
