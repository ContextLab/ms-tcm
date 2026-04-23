#!/usr/bin/env python3
"""Check paper/main.tex body for v1-retired strings and required v6 strings.

Per specs/002-ms-tcm-v6-hcmr/contracts/paper-consistency.md §2.4, §2.5, §4,
§8, §9 and FR-049.

Exit codes (contract §9):
    0: all checks pass
    1: one or more banned strings found in main.tex body
    2: one or more required strings missing from main.tex body
    3: Cornell & Zhang 2025 bib entry missing or malformed
    4: notes/v6_migration.md missing required sections

Inputs that are grepped (latex comments are stripped; a migration
supplement note in ``paper/supplement.tex`` §S0 is exempt from the
banned-list check per contract §3):

    paper/main.tex body (comments stripped)
    paper/local.bib          (for CornellZhang2025 entry)
    paper/CDL-bibliography/cdl.bib (fallback)
    notes/v6_migration.md   (per contract §8)

Run from the repository root:

    python scripts/check_paper_consistency.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


# Contract §2.4 banned strings. Match as plain substrings against the
# comment-stripped main.tex body. v1 mathematical symbols + composite
# terminology + §4.4 numerical anchors.
_BANNED = [
    r"\beta_G",
    r"\beta_S",
    r"\beta_{G}",
    r"\beta_{S}",
    r"w_G",
    r"w_S",
    r"w_{G}",
    r"w_{S}",
    r"w^{ret}_G",
    r"w^{ret}_S",
    r"\wGret",
    r"\wSret",
    "0.866",
    "0.80532",
    "0.806",
    "frozen storyline",
    r"w_{\mathrm{global}} + w_{\mathrm{storyline}}",
    r"w\_global + w\_storyline",
    "composite similarity",
]

# Contract §2.5 required substrings.
_REQUIRED = [
    r"CornellZhang2025",
    r"hierarchical context",
    r"\beta_{\mathrm{enc}}",
    r"\beta_{\mathrm{story}}",
    r"\lambda",
    r"M^{\mathrm{IC}}",
    r"M^{\mathrm{SC}}",
]


def _strip_latex_comments(src: str) -> str:
    """Remove everything after a non-escaped ``%`` on each line.

    LaTeX comment syntax: ``%`` begins a comment unless it is escaped
    (``\\%``). A literal ``%`` in math mode is still a comment (TeX
    does not distinguish), so a line-by-line split suffices here.
    """
    out_lines: list[str] = []
    for line in src.splitlines():
        # Walk char-by-char and break at the first non-escaped ``%``.
        idx = 0
        n = len(line)
        cut = n
        while idx < n:
            ch = line[idx]
            if ch == "\\":
                idx += 2
                continue
            if ch == "%":
                cut = idx
                break
            idx += 1
        out_lines.append(line[:cut])
    return "\n".join(out_lines)


def _report(banned_hits, missing_required, bib_problem, migration_problem) -> int:
    payload = {
        "banned_hits": banned_hits,
        "missing_required": missing_required,
        "bib_problem": bib_problem,
        "migration_problem": migration_problem,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if banned_hits:
        print("FAIL: banned strings found in main.tex body.", file=sys.stderr)
        for hit in banned_hits:
            print(f"  - {hit}", file=sys.stderr)
        return 1
    if missing_required:
        print("FAIL: required strings missing from main.tex body.", file=sys.stderr)
        for s in missing_required:
            print(f"  - {s}", file=sys.stderr)
        return 2
    if bib_problem:
        print(f"FAIL: bib entry issue: {bib_problem}", file=sys.stderr)
        return 3
    if migration_problem:
        print(f"FAIL: v6_migration.md issue: {migration_problem}", file=sys.stderr)
        return 4
    return 0


def _check_bib(repo_root: Path) -> str | None:
    """Return None if bib is ok, else a human-readable error string.

    Searches paper/local.bib then paper/CDL-bibliography/cdl.bib
    (cumulative — either may carry the CornellZhang2025 entry per
    contract §4). Verifies author / journal / year / (doi OR url/note)
    presence per contract §2.2 + T055b. ``note`` is accepted in lieu of
    ``doi`` / ``url`` when it is non-empty and references the preprint
    PDF (see research.md R5 bibtex template).
    """
    candidate_bibs = [
        repo_root / "paper" / "local.bib",
        repo_root / "paper" / "CDL-bibliography" / "cdl.bib",
    ]
    src_parts: list[str] = []
    for path in candidate_bibs:
        if path.exists():
            src_parts.append(path.read_text())
    src = "\n".join(src_parts)
    if not src:
        return "no bib files found"
    # Locate the CornellZhang2025 entry.
    m = re.search(
        r"@\w+\s*\{\s*CornellZhang2025\s*,"
        r"(?P<body>.*?)\n\}\s*", src, re.DOTALL,
    )
    if not m:
        return "CornellZhang2025 entry not found in bib"
    body = m.group("body")
    # Required fields — each must appear as a non-empty value.
    required_fields = ["author", "title", "journal", "year"]
    for field in required_fields:
        fm = re.search(
            rf"\b{field}\s*=\s*[{{\"]([^}}\"]*)[}}\"]",
            body, re.IGNORECASE,
        )
        if not fm or not fm.group(1).strip():
            return f"CornellZhang2025 bib entry missing '{field}'"
    # At least one of doi / url / note must be non-empty.
    has_pointer = False
    for field in ("doi", "url", "note"):
        fm = re.search(
            rf"\b{field}\s*=\s*[{{\"]([^}}\"]*)[}}\"]",
            body, re.IGNORECASE,
        )
        if fm and fm.group(1).strip():
            has_pointer = True
            break
    if not has_pointer:
        return "CornellZhang2025 bib entry must include a doi, url, or note field"
    # Check author contains the two expected surnames.
    am = re.search(
        r"\bauthor\s*=\s*[{\"]([^}\"]+)[}\"]",
        body, re.IGNORECASE,
    )
    if am:
        authors_str = am.group(1)
        if "Cornell" not in authors_str or "Zhang" not in authors_str:
            return "CornellZhang2025 author field missing Cornell or Zhang"
    # Check journal is Psychological Review.
    jm = re.search(
        r"\bjournal\s*=\s*[{\"]([^}\"]+)[}\"]",
        body, re.IGNORECASE,
    )
    if jm and "Psychological Review" not in jm.group(1):
        return "CornellZhang2025 journal is not Psychological Review"
    # Check year is 2025.
    ym = re.search(
        r"\byear\s*=\s*[{\"]?(\d{4})[}\"]?", body, re.IGNORECASE,
    )
    if ym and ym.group(1) != "2025":
        return "CornellZhang2025 year must be 2025"
    return None


def _check_migration_doc(repo_root: Path) -> str | None:
    """Per contract §8 + task T078d: assert required sections exist."""
    path = repo_root / "notes" / "v6_migration.md"
    if not path.exists():
        return "notes/v6_migration.md missing"
    text = path.read_text().lower()
    required_substrings = ["symbol mapping", "deletion manifest", "rationale"]
    missing = [s for s in required_substrings if s not in text]
    if missing:
        return f"notes/v6_migration.md missing sections: {', '.join(missing)}"
    return None


def check_paper(repo_root: Path) -> int:
    main_tex = repo_root / "paper" / "main.tex"
    if not main_tex.exists():
        print(f"error: paper/main.tex not found at {main_tex}", file=sys.stderr)
        return 2
    src = main_tex.read_text()
    body = _strip_latex_comments(src)

    banned_hits = [s for s in _BANNED if s in body]
    missing_required = [s for s in _REQUIRED if s not in body]

    bib_problem = _check_bib(repo_root)
    migration_problem = _check_migration_doc(repo_root)

    return _report(banned_hits, missing_required, bib_problem, migration_problem)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=_REPO_ROOT,
        help="Repository root (default: script's parent-of-parent).",
    )
    args = parser.parse_args(argv)
    return check_paper(args.repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
