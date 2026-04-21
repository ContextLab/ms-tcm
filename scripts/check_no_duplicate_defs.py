"""Constitution II guard: no duplicate function/class definitions.

Scans `code/ms_tcm/` for top-level `def` and `class` names and fails if any
name appears in more than one file. Also scans `code/notebooks/` for top-level
`def` and `class` names in code cells and fails if any name exists in both
the package and a notebook (notebooks must import, not redefine).

Run manually (used by T054): ``python scripts/check_no_duplicate_defs.py``.
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "code" / "ms_tcm"
NOTEBOOK_DIR = ROOT / "code" / "notebooks"


def _top_level_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _scan_package() -> tuple[dict[str, list[str]], set[str]]:
    by_name: dict[str, list[str]] = defaultdict(list)
    all_names: set[str] = set()
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        names = _top_level_names(path.read_text())
        for n in names:
            by_name[n].append(str(path.relative_to(ROOT)))
            all_names.add(n)
    return by_name, all_names


def _scan_notebooks() -> dict[str, list[str]]:
    by_name: dict[str, list[str]] = defaultdict(list)
    for path in sorted(NOTEBOOK_DIR.rglob("*.ipynb")):
        nb = json.loads(path.read_text())
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", [])
            if isinstance(source, list):
                source = "".join(source)
            names = _top_level_names(source)
            for n in names:
                by_name[n].append(str(path.relative_to(ROOT)))
    return by_name


def main() -> int:
    pkg_by_name, pkg_names = _scan_package()
    violations: list[str] = []

    for name, files in pkg_by_name.items():
        if len(files) > 1:
            violations.append(
                f"package duplicate: {name!r} defined in {files}"
            )

    nb_by_name = _scan_notebooks()
    for name, files in nb_by_name.items():
        if name in pkg_names:
            violations.append(
                f"notebook redefines package symbol: {name!r} in {files} "
                f"(also defined in {pkg_by_name[name]})"
            )

    if violations:
        print("Constitution II violations:\n  " + "\n  ".join(violations))
        return 1
    print(
        f"OK: {sum(len(v) for v in pkg_by_name.values())} top-level names across "
        f"{len({f for fs in pkg_by_name.values() for f in fs})} package files; "
        f"no duplicates; no notebook redefinitions."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
