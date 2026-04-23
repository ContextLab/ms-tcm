# Contract: Paper + Documentation Consistency (FR-040 through FR-049)

**Feature**: 002-ms-tcm-v6-hcmr
**Enforced by**: `scripts/check_paper_consistency.py`

## 1. Scope

This contract specifies *what* the paper, `CLAUDE.md`, and the three READMEs must say (and not say) after the v6 rewrite, and *how* the CI job verifies compliance.

## 2. `paper/main.tex` — mandatory edits

### 2.1 Section rewrites

- **§3 (Model)**: rewritten to match `notes/two_level_cmr_v6.pdf` §1–§3. Must contain the equations labeled Eq 1 through Eq 9 of v6 (item drift, storyline drift, ΔM^IC, event-boundary sync, ΔM^SC, storyline-return reinstatement, boundary-sync at switch, retrieval activation, retrieval softmax). Figure 1 (concept diagram) regenerated from `code/figures/make_fig_model.py`.
- **§4 (Derivation)**: rewritten to explain the grouped-bridge equivalence via v6 mechanisms (item-context boundary sync + λ reinstatement), NOT via v1 `w_S >> w_G` reweighting. §4.4 numerical anchor removed entirely.
- **New §1.5-equivalent** (title TBD during implementation): one or two paragraphs discussing the choice of M^FC_pre (identity vs. USE embeddings), per v6 §1.5.

### 2.2 Required citations

At least one `\citep{CornellZhang2025}` or `\citet{CornellZhang2025}` in §3 or §Methods. The sentence must explicitly identify MS-TCM as extending C&Z 2025 by adding exactly one mechanism (λ, storyline-return reinstatement at encoding).

### 2.3 Parameter inventory

A table in §Methods listing the v6 parameters with their symbols, C&Z 2025 Table 1 starting values (for the 6 inherited), v6 §5 starting value (for λ), and roles. No mention of v1-retired parameters.

### 2.4 Banned literal strings

The following strings **must not appear** in `main.tex` body (excluding `%`-style LaTeX comments, `\bibliography{}` commands, and the migration note if it's in `paper/supplement.tex`):

| Banned string | Why |
|-|-|
| `\beta_G`, `\beta_S`, `\beta_{G}`, `\beta_{S}` | v1 symbols |
| `w_G`, `w_S`, `w_{G}`, `w_{S}` | v1 encoding weights |
| `w^{ret}_G`, `w^{ret}_S` | v1 retrieval reweighting |
| `\wGret`, `\wSret` | macros defined in v1 preamble |
| `0.866`, `0.80532`, `0.806` | v1 §4.4 anchor numerics |
| `frozen storyline` | v1 inactive-storyline language |
| `w_{\mathrm{global}} + w_{\mathrm{storyline}}`, `w\_global + w\_storyline` | v1 sum-to-one constraint |
| `composite similarity` (verbatim phrase) | v1 §3.3 terminology |

### 2.5 Required literal strings

The following strings **must appear** in `main.tex` body:

| Required string | Approximate location |
|-|-|
| `\cite{CornellZhang2025}` or `\citet{CornellZhang2025}` or `\citep{CornellZhang2025}` | §3 or §Methods |
| `hierarchical context` | §3 (v6 framing) |
| `\beta_{\mathrm{enc}}`, `\beta_{\mathrm{story}}` | §Methods parameter table |
| `\lambda` with text identifying it as storyline-return reinstatement | §3 or §4 |
| `M^{\mathrm{IC}}`, `M^{\mathrm{SC}}` | §3 |

## 3. `paper/supplement.tex` — optional migration note

If `notes/v6_migration.md` is not authored, the v1→v6 migration note goes into `paper/supplement.tex` instead. The consistency checker treats the migration note (wherever it lives) as *exempt* from the banned-string list above — it is expected to discuss v1 symbols for reference.

## 4. Bibliography

One of the following must be true:

- `paper/CDL-bibliography/cdl.bib` contains a `@article{CornellZhang2025, ...}` entry.
- `paper/local.bib` contains a `@article{CornellZhang2025, ...}` entry AND `paper/main.tex` includes `\bibliography{CDL-bibliography/cdl,local}` (or equivalent multi-bib include).

The entry must have: author, title, journal, year, and either a DOI or a preprint URL.

## 5. `CLAUDE.md` — mandatory edits

- The "Project" section's canonical-spec pointer changes from `notes/ms-tcm.pdf` to `notes/two_level_cmr_v6.pdf`.
- The parameter-regimes list changes from v1 symbols (β_G, β_S, w_G, w_S, γ, λ) to v6 symbols (β_enc, β_story, γ_fc, k, λ, β_rec, ε_d).
- The §4.4 numerical-sanity-check paragraph is replaced with: "Numerical-accuracy anchor: the `--standard-tcm` reduction (λ=0, one storyline) must pass `code/tests/test_hcmr_standard_tcm.py` and the Layer 1 shape assertions of `code/tests/test_behavioral_regression.py` before any MS-TCM claim is trusted."
- The common-commands section gains a one-liner for `ms-tcm benchmark` and notes the Tier 1 < 120 s target.

## 6. `README.md`, `code/README.md`, `data/README.md` — mandatory edits

- `README.md`: replace the CDL-template "Paper title" placeholder with the project-specific title ("A Multi-Stream Temporal Context Model for Narrative Memory"). Add one sentence summarizing the Tier 1 performance claim. Point at `specs/002-ms-tcm-v6-hcmr/` as the active feature spec.
- `code/README.md`: update the CLI examples to use v6 parameter names; note the new `benchmark` subcommand; add a one-line pointer to `specs/002-ms-tcm-v6-hcmr/contracts/cli.md`.
- `data/README.md`: note the new `data/processed/reference_curves/` directory and its regeneration command. FRFR-category documentation is unchanged.

## 7. `specs/001-ms-tcm-impl/plan.md` — forward pointer

A one-line banner at the top of 001's plan.md:

```
**SUPERSEDED** by [specs/002-ms-tcm-v6-hcmr/plan.md](../002-ms-tcm-v6-hcmr/plan.md). The v1 MS-TCM math was retired on 2026-04-23; see [notes/v6_migration.md](../../notes/v6_migration.md) for the symbol mapping.
```

## 8. `notes/v6_migration.md` — new file

Contents:

1. **Symbol mapping table**: v1 symbol → v6 symbol, with caveats.
2. **Deletion manifest**: enumerated list of v1-specific files and tests removed per FR-013.
3. **Rationale**: 3–5 paragraphs summarizing why v6 replaced v1 (behavioral phenomena failure, alignment with Cornell & Zhang 2025).

## 9. CI gate: `scripts/check_paper_consistency.py`

Runs in CI on every commit touching `paper/`, `CLAUDE.md`, or `README.md`/`code/README.md`/`data/README.md`.

Exit codes:
- 0: all checks pass.
- 1: one or more banned strings found in `main.tex` body.
- 2: one or more required strings missing from `main.tex` body.
- 3: Cornell & Zhang 2025 bib entry missing or malformed.
- 4: `CLAUDE.md` or a README not updated (based on keyword grep).

Output: JSON summary to stdout; human-readable error pointers to stderr.

## 10. Figure regeneration (FR-048)

Each figure in `paper/figs/` has a generator script in `code/figures/`. Any figure whose content depends on v1 math (v1 §4.4 anchor plot, if present) is regenerated or deleted. The consistency checker does not enforce figure content directly; instead, it greps the LaTeX `\includegraphics{}` paths and asserts the referenced PDFs have an `mtime` later than the last `v1-math-dependent` git commit touching the generator script.
