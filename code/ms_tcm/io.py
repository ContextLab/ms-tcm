"""I/O helpers: SHA-256 hashing and deterministic JSON writer.

See ``specs/001-ms-tcm-impl/contracts/dataset-schema.md`` §4–§5.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the hex SHA-256 digest of a file, streaming 64 KiB at a time."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_sorted(obj: Any, path: str | os.PathLike[str]) -> None:
    """Write JSON with sorted keys, two-space indent, trailing newline.

    These settings make the file byte-stable given the same logical content,
    which is what the manifest contract requires (FR-009).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")


def read_json(path: str | os.PathLike[str]) -> Any:
    """Read a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
