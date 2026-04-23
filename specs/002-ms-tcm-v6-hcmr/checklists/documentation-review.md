# Documentation Review Checklist (A15 / T078c)

**Feature**: 002-ms-tcm-v6-hcmr
**Reviewer**: Jeremy Manning
**Date**: 2026-04-23
**Branch**: 002-ms-tcm-v6-hcmr

Filled in as part of the Milestone 7 polish pass; referenced by the PR
body per T079. Each box is ticked only after the referenced file has
been read end-to-end, compared against the requirement, and found
compliant.

## CLAUDE.md

- [x] **Canonical spec pointer**: `notes/two_level_cmr_v6.pdf` replaces
      the v1 `notes/ms-tcm.pdf` pointer in the Project section
      (paper-consistency §5).
- [x] **v6 parameter regimes**: the parameter-regimes list uses
      `β_enc`, `β_story`, `γ_fc`, `k`, `λ`, `β_rec`, `ε_d` (no `β_G`,
      `β_S`, `w_G`, `w_S`).
- [x] **§4.4 anchor paragraph replaced**: the v1 §4.4 rounding-cascade
      sanity check is replaced with the `--standard-tcm` + behavioural
      regression anchor per paper-consistency §5.
- [x] **Benchmark one-liner added**: the Common-commands / Modeling
      section now references `scripts/benchmark_fit.py` and the
      Tier 1 < 120 s target.

## README.md

- [x] **Project title updated**: "A Multi-Stream Temporal Context
      Model for Narrative Memory" replaces the v1 placeholder title
      (paper-consistency §6).
- [x] **v6 framing**: MS-TCM is described as hierarchical CMR
      (Cornell and Zhang 2025) plus one new mechanism (λ).
- [x] **Performance claim**: one sentence describes the Tier 1
      < 2 minutes target.
- [x] **Active feature spec link**: points to
      `specs/002-ms-tcm-v6-hcmr/`.

## code/README.md

- [x] **v6 public API**: `HierarchicalCMRModel` (not `MSTCMModel`),
      `MFCPreMatrix` / `IdentityPreMatrix` / `EmbeddingPreMatrix`,
      `benchmark_fit` are listed.
- [x] **Forward pointer** to
      `specs/002-ms-tcm-v6-hcmr/contracts/cli.md` instead of the
      001-feature contract.
- [x] **New CLI section**: the three subcommands (`validate`, `fit`,
      `benchmark`) are shown with example invocations.
- [x] **Figure scripts**: `make_fig_behavioral.py` listed alongside
      `make_fig_model.py` (both regenerated in Phase 5).
- [x] **Test invocation examples**: distinguish `-m "not slow"` from
      `-m "slow or not slow"`.

## data/README.md

- [x] **`data/processed/reference_curves/`** documented with its
      regeneration command (`scripts/build_reference_curves.py`).
- [x] **`data/processed/benchmarks/benchmark_log.csv`** documented
      with its role (FR-031 / SC-009).
- [x] **FRFR-category byte-identity** note preserved (FR-061).

## specs/001-ms-tcm-impl/plan.md — SUPERSEDED banner

- [x] A one-line banner at the top points to
      `specs/002-ms-tcm-v6-hcmr/plan.md` and `notes/v6_migration.md`
      (paper-consistency §7).

## specs/001-ms-tcm-impl/contracts/

- [x] Each file either updated to v6 API or explicitly forward-pointed
      to `specs/002-ms-tcm-v6-hcmr/contracts/` (FR-046). Carried over
      from Phase 1 git history.

## notes/v6_migration.md

- [x] **Symbol mapping table** present (v1 → v6, with Python
      identifier column and caveats).
- [x] **Deletion manifest** present (enumerated list of retired files
      and tests per FR-013).
- [x] **Rationale** section present (3–5 paragraphs explaining why
      v6 replaced v1).
- [x] **Reinterpret-v1-fits** guidance present.

## paper/main.tex

- [x] **§2/§3 (Model)** rewritten to match v6 §1–§3 (two context
      levels, M^IC / M^SC, λ reinstatement). Contains the explicit
      sentence "MS-TCM adds exactly one mechanism to Cornell and
      Zhang's (2025) hierarchical CMR" (paper-consistency §2.2).
- [x] **§4 (Derivation)** rewritten to the v6 narrative. §4.4
      numerical anchor removed entirely.
- [x] **§Methods parameter table**: v6 symbols with C&Z 2025 Table 1
      starting-value citations; no v1 parameters.
- [x] **Cornell and Zhang (2025) cited** in §2 and §Methods
      (paper-consistency §2.2).
- [x] **M^FC_pre subsection** added covering the identity-vs-embedding
      choice (§\ref{sec:preexp}); cites Polyn et al 2009, Morton and
      Polyn 2016, Heusser et al 2021, Xu et al 2024 (paper-consistency
      §2.1 new-§1.5).

## paper/local.bib (or CDL submodule)

- [x] **CornellZhang2025** entry present with author, title, journal
      (Psychological Review), year (2025), and a note/url pointing at
      the preprint (paper-consistency §4; T055b-verified).
- [x] `scripts/check_paper_consistency.py` exits 0 (the SC-008
      sub-check).

## Build + checks

- [x] `cd paper && ./compile.sh` produces `main.pdf` + `supplement.pdf`
      without undefined-reference warnings.
- [x] `scripts/check_paper_consistency.py` exits 0.
- [x] `scripts/check_figure_provenance.py` exits 0.
- [x] `scripts/check_no_duplicate_defs.py` exits 0.

---

All boxes ticked → ready for PR.
