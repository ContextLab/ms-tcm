#!/usr/bin/env python3
"""Check that every paper figure has a generator script or notebook cell.

Per T078b / A4 resolution:
  - Parse paper/main.tex (and paper/supplement.tex) for every
    ``\\includegraphics{PATH}`` reference.
  - For each referenced PDF under paper/figs/, verify that either
    (a) a generator script exists under code/figures/ whose filename
        maps to the figure name (``fig_X.pdf`` -> ``make_fig_X.py``),
        or
    (b) a notebook under code/notebooks/ or a script under
        code/figures/ references the figure's full basename as a
        write target.

Exit codes:
    0: every figure has a regeneration path
    1: one or more figures are orphans
    2: paper/main.tex missing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _collect_includegraphics(tex_paths: list[Path]) -> list[str]:
    """Return all includegraphics paths across the supplied .tex files."""
    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    paths: list[str] = []
    for p in tex_paths:
        if not p.exists():
            continue
        src = p.read_text()
        for m in pattern.finditer(src):
            paths.append(m.group(1))
    return paths


def _has_generator(fig_basename: str, figures_dir: Path,
                   notebooks_dir: Path) -> bool:
    """Return True iff a generator script or notebook can produce this figure.

    Rule 1: ``figures/make_<basename>.py`` exists.
    Rule 2: Any ``figures/*.py`` or ``notebooks/*.ipynb`` contains the
            basename as a literal substring (as a write-target path).
    """
    # Rule 1: convention-based generator script.
    script = figures_dir / f"make_{fig_basename}.py"
    if script.exists():
        return True
    # Rule 2: fallback full-file search.
    for candidate_dir, suffix in (
        (figures_dir, ".py"),
        (notebooks_dir, ".ipynb"),
    ):
        if not candidate_dir.exists():
            continue
        for f in candidate_dir.rglob(f"*{suffix}"):
            try:
                if fig_basename in f.read_text(errors="replace"):
                    return True
            except Exception:
                continue
    return False


def check_provenance(repo_root: Path) -> int:
    main_tex = repo_root / "paper" / "main.tex"
    supp_tex = repo_root / "paper" / "supplement.tex"
    figures_dir = repo_root / "code" / "figures"
    notebooks_dir = repo_root / "code" / "notebooks"

    if not main_tex.exists():
        print(f"error: {main_tex} not found", file=sys.stderr)
        return 2

    includes = _collect_includegraphics([main_tex, supp_tex])
    orphans: list[str] = []
    for inc in includes:
        # Extract the basename (without extension), ignoring any leading
        # directory prefix used for includegraphics.
        name = Path(inc).name
        # PDF/PNG extension may or may not be in the includegraphics arg;
        # strip any trailing .pdf / .png / .jpg / .eps for the generator lookup.
        basename = re.sub(r"\.(pdf|png|jpg|jpeg|eps)$", "", name,
                          flags=re.IGNORECASE)
        if not _has_generator(basename, figures_dir, notebooks_dir):
            orphans.append(inc)

    payload = {
        "includegraphics": includes,
        "orphans": orphans,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if orphans:
        print(
            f"FAIL: {len(orphans)} figure(s) have no regeneration path:",
            file=sys.stderr,
        )
        for o in orphans:
            print(f"  - {o}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=_REPO_ROOT,
        help="Repository root (default: script's parent-of-parent).",
    )
    args = parser.parse_args(argv)
    return check_provenance(args.repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
