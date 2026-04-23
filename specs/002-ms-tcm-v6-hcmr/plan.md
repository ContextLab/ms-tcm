# Implementation Plan: MS-TCM v6 (Hierarchical CMR) Rewrite + Fast Inference + Paper/Docs Sync

**Branch**: `002-ms-tcm-v6-hcmr` | **Date**: 2026-04-23 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-ms-tcm-v6-hcmr/spec.md`

## Summary

Replace the v1 MS-TCM code (`code/ms_tcm/`) with a Cornell & Zhang 2025-style hierarchical CMR plus the v6 λ storyline-return mechanism (exactly one new parameter over C&Z 2025). Add first-class behavioral regression tests for serial position curve, probability of first recall, and lag-CRP (two-layer policy: qualitative shape + ≤ 20 % relative-error per-bin against FRFR-category empirical curves, MS-TCM no-worse-than `--standard-tcm`). Cut fit wall-clock from ~600 s to < 120 s (Tier 1, mandatory) via vectorized encoding, cached pre-experimental matrix, and multiprocessing-parallelized bootstrap, with an optional Tier 2 JAX path (< 30 s, float64 default / float32 opt-in). Propagate the v6 notation through `paper/main.tex`, `CLAUDE.md`, the three READMEs, and all existing `specs/001-ms-tcm-impl/` contracts, and add Cornell & Zhang 2025 to the bibliography. The v1 code (composite.py, the v1 parts of context.py/mechanisms.py/params.py, §4.4 tests, v1 notebook cells) is destructively deleted per the Q1 clarification.

## Technical Context

**Language/Version**: Python 3.11 (pinned in existing `pyproject.toml` and `Dockerfile` — no change).

**Primary Dependencies** (unchanged unless called out):

- `numpy>=1.26`, `scipy>=1.11`, `pyarrow>=14`, `pandas>=2.1`, `h5py>=3.10` — inherited.
- `pytest>=7`, `pytest-cov` — inherited; add `pytest-xdist` for parallel test execution of the behavioral regression suite.
- **NEW for Tier 1**: Python standard library `multiprocessing.Pool` — no new deps.
- **NEW for Tier 2 (optional)**: `jax>=0.4.28`, `jaxlib>=0.4.28`, `optax>=0.2.0`, added to an optional extras group `[project.optional-dependencies].jax` so the default install path is unchanged. Tier 2 is gated by backend selection, with precedence [A8 resolved]: `--backend` CLI flag > `MS_TCM_BACKEND` environment variable > Tier 1 default. The pyproject-toml setting mentioned in earlier drafts is DROPPED (redundant; env var + CLI flag suffice). If JAX backend is requested but the `jax` extras group is not installed, the CLI logs a warning and auto-falls-back to Tier 1 (A6; tested by T045b).
- **NEW**: `matplotlib>=3.8` made explicit in `[project.optional-dependencies].figures`.
- **NEW**: a hand-rolled `scripts/benchmark_fit.py` emits JSON (chosen over `pytest-benchmark` because the CI job needs a stable machine-readable format that survives code-path reorganization).

**Storage**: Local filesystem. `data/raw/frfr_category/` is byte-identical (no schema changes). A new `data/processed/reference_curves/` directory receives the SPC/pFR/lag-CRP reference curves for behavioral regression tests (Q2 Layer 2 targets). A new `data/processed/benchmarks/benchmark_log.csv` accumulates benchmark records (SC-009).

**Testing**:

- `pytest` with deterministic seeds; 1e-10 absolute tolerance for Tier 1 + Tier 2-float64; 1e-8 for Tier 2-float32.
- New test module `code/tests/test_behavioral_regression.py` implementing the Q2 two-layer policy.
- New test module `code/tests/test_preexp.py` for the pluggable M^FC_pre API.
- New test module `code/tests/test_param_defaults.py` for C&Z 2025 Table 1 starting-value parity.
- Existing `code/tests/test_parameter_recovery.py` rewritten in place on the v6 parameter set (β_enc, β_story, γ_fc, k, λ, β_rec, ε_d).
- Deleted: `test_section_4_4_numerical_anchor` (v1 §4.4), any test asserting `w_global + w_storyline == 1`, `test_similarity.py` (wrapper subsumed into retrieval.py).

**Target Platform**: macOS, Ubuntu, Windows (Constitution Principle IV). Existing `.github/workflows/ci.yml` matrix extended with a second dimension for `MS_TCM_JAX_DTYPE={float64, float32}` when the JAX extra is installed.

**Project Type**: Scientific Python library + one-off reformat script + LaTeX paper + Jupyter notebooks. Unchanged from 001.

**Performance Goals**:

- Tier 1: `ms-tcm fit data/raw/frfr_category --n-bootstraps 1000 --n-restarts 5 --seed 42` completes in < 120 s on CI hardware (GitHub-hosted `macos-latest` / `ubuntu-latest`, 2-core 7-GB RAM). Baseline: ~600 s. Target speedup ≥ 5×.
- Tier 2 (optional): same command < 30 s. Target speedup ≥ 20×.
- Behavioral regression suite: < 3 min total on CI.
- `ms-tcm validate data/raw/frfr_category`: < 10 s (unchanged from 001).

**Constraints**:

- Cross-platform numerical tolerance: 1e-10 on Tier 1 and Tier 2-float64 paths; 1e-8 on Tier 2-float32 opt-in path.
- Bit-exact preservation of `data/raw/frfr_category/` (manifest SHA-256s unchanged).
- Zero duplicate function definitions (`scripts/check_no_duplicate_defs.py` stays green).
- No mocks, no stubs, no silent fallbacks (Constitution I + global `CLAUDE.md` rule).
- Symbols mirror v6 notation (β_enc, β_story, γ_fc, k, λ, β_rec, ε_d), not v1 (β_G, β_S, w_G, w_S). Constitution III.
- Paper `main.pdf` and `supplement.pdf` build without undefined-reference warnings after the v6 rewrite.

**Scale/Scope**: FRFR-category (unchanged): 30 participants × 16 lists × 16 words, 4 categories per list (≡ 4 storylines). 7680 presented rows, 5215 recalled rows. Fit has 7 parameters (β_enc, β_story, γ_fc, k, λ, β_rec, ε_d) in the free-recall paradigm, 5 in the cued-recall paradigm (β_enc, β_story, γ_fc, k, λ).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The constitution (`v1.0.0`, ratified 2026-04-20) defines four core principles. Evaluation for this feature:

| Principle | Gate | Status | Evidence |
|-|-|-|-|
| **I. Accuracy (NON-NEGOTIABLE)** | Every numerical function has a unit test with a hand-derivable expected answer; every citation traceable; no mocks; model claims trace to a v6 notes section, a C&Z 2025 equation, or the FRFR-category manifest. | **Pass** | FR-009 pins parameter starting values to C&Z Table 1; FR-020 pins behavioral regression targets to FRFR-category empirical curves; `test_param_defaults.py` and `test_behavioral_regression.py` enforce both at CI time. The constitution's "§4.4 anchor" requirement is explicitly superseded by v6 (which retires §4.4); the migration note in `notes/v6_migration.md` (FR-047) records the redirection. |
| **II. Single Source of Truth** | Every function defined once, imported elsewhere. No `_v2` siblings. Derived artifacts (figures, reference curves, benchmark log) regenerable from committed scripts. | **Pass** | All new code lives under `code/ms_tcm/`; FR-013 destructively removes v1 modules (no `*_v1.py` siblings); `scripts/check_no_duplicate_defs.py` stays green; reference curves cached under `data/processed/reference_curves/` are regenerable from a committed script (`scripts/build_reference_curves.py`). |
| **III. Clarity** | Symbols match v6 notation on first use; jargon defined; names describe behavior not provenance. | **Pass** | FR-044 updates `CLAUDE.md` parameter regimes to v6 symbols; paper §3 / §4 / Methods rewritten to v6 (FR-040 / FR-041 / FR-042); module names in `code/ms_tcm/` describe hierarchical-CMR roles (`hcmr.py`, `retrieval.py`, `matrices.py`, `preexp.py`) not "new"/"v2"/"fixed". |
| **IV. Reproducibility (NON-NEGOTIABLE)** | Cross-platform bit-exact dataset hashes; derived outputs within 1e-10 (Tier 1, Tier 2-float64) or 1e-8 (Tier 2-float32); seeds explicit; scripts idempotent. | **Pass** | FR-023 pins tolerances; FR-061 preserves `data/raw/frfr_category/`; the existing `scripts/reformat_frfr_category.py` is untouched; `scripts/benchmark_fit.py` is idempotent (re-runs overwrite the same benchmark record deterministically given `--seed`); new `data/processed/reference_curves/` entries have SHA-256s in a companion `manifest.json`. |

**Additional constitution-level concerns:**

- Constitution text refers to `notes/ms-tcm.pdf` as "the canonical specification" (paragraph before Core Principles). v6 changes this. Two resolution options:
    - **(A)** Amend the constitution in this feature's first commit (version bump 1.0.0 → 1.1.0) to point at `notes/two_level_cmr_v6.pdf`.
    - **(B)** Leave the constitution text alone, interpret `notes/ms-tcm.pdf` historically, and record the redirection in `notes/v6_migration.md` + `CLAUDE.md`.
  **Decision: (A)** — the constitution is MINOR-bumped (1.0.0 → 1.1.0) to recognize v6 as the canonical spec. This is explicitly within the constitution's own amendment procedure ("expanded guidance within an existing section"). The bump is captured in `research.md` section R0.
- Constitution I cites the §4.4 numerical anchor (0.866 / 0.806) as "required regression anchors for any MS-TCM similarity code". The amendment above also removes this specific reference (the anchor math no longer holds under v6). A replacement numerical-accuracy anchor is introduced: the v6 `--standard-tcm` reduction is asserted to match Cornell & Zhang 2025 Table 1 parameter defaults and to reproduce the Kahana et al. 2002 free-recall phenomena in the Q2 Layer 1 shape-assertion sense. This is recorded in the amended constitution.

**Post-Phase-1 re-check**: same four principles; data-model and contracts below preserve all four. No new violations. Pass.

## Project Structure

### Documentation (this feature)

```text
specs/002-ms-tcm-v6-hcmr/
├── plan.md                       # THIS FILE (/speckit.plan output)
├── spec.md                       # Feature specification with Session 2026-04-23 clarifications
├── research.md                   # Phase 0 output (this run)
├── data-model.md                 # Phase 1 output (this run)
├── quickstart.md                 # Phase 1 output (this run)
├── contracts/
│   ├── model-api.md              # Public Python surface of `ms_tcm` (v6)
│   ├── cli.md                    # `ms-tcm validate|run|fit|benchmark` CLI contract
│   ├── fitter.md                 # MLE + bootstrap-CI + performance-tier contract
│   ├── dataset-schema.md         # Re-published from 001 (unchanged) with a forward-pointer note
│   ├── regression-tests.md       # Behavioral regression (SPC/pFR/lag-CRP) contract (new)
│   └── paper-consistency.md      # Paper/docs sync + bib entry contract (new)
├── checklists/
│   └── requirements.md           # /speckit.specify quality checklist (already written)
└── tasks.md                      # Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

```text
code/
├── ms_tcm/                       # Python package (single source of truth, Constitution II)
│   ├── __init__.py               # Re-exports public API (v6 names)
│   │
│   │  — Core math (v6 hierarchical CMR) —
│   ├── hcmr.py                   # HierarchicalCMRModel orchestrator (replaces model.py)
│   ├── drift.py                  # Item-level and storyline-level drift (v6 §2.1, replaces context.py)
│   ├── boundaries.py             # Event-boundary sync + storyline-switch caching + storyline-return reinstatement (v6 §2.2-2.4)
│   ├── matrices.py               # M^IC, M^SC association matrices + updates
│   ├── preexp.py                 # MFCPreMatrix interface: identity default, embeddings hook (v6 §1.5, FR-007)
│   ├── retrieval.py              # a = (M^IC)^T c^item; softmax with gain k; β_rec drift; ε_d stopping rule (free-recall only) — FR-005, FR-006
│   ├── params.py                 # ModelParameters (v6 schema: β_enc, β_story, γ_fc, k, λ, β_rec, ε_d) with C&Z Table 1 defaults (FR-009)
│   │
│   │  — Data + likelihood + fit (mostly inherited from 001) —
│   ├── features.py               # Vectorized multi-hot feature encoder (Tier 1: no iterrows — FR-030)
│   ├── schema.py                 # Unchanged from 001
│   ├── dataset.py                # Unchanged from 001
│   ├── frfr.py                   # Unchanged from 001
│   ├── likelihood.py             # Rewritten around v6 retrieval route
│   ├── fit.py                    # Rewritten reparameterization for v6 parameter set
│   ├── bootstrap.py              # Multiprocessing-parallelized bootstrap (Tier 1 — FR-030)
│   ├── benchmark.py              # New: instrument `fit` wall-clock + memory, emit machine-readable JSON (FR-031)
│   ├── io.py                     # Unchanged from 001
│   └── cli.py                    # Extended: `--pre-context`, `--backend`, `--jax-dtype`, `benchmark` subcommand
│   │
│   │  — Optional Tier 2 JAX backend —
│   └── jax_backend/              # Only imported when MS_TCM_BACKEND=jax (FR-032)
│       ├── __init__.py
│       ├── hcmr_jax.py           # JIT-compiled likelihood + autodiff
│       └── fit_jax.py            # Optax-based optimizer (replaces scipy L-BFGS-B when enabled)
│
├── analyses/                     # 001's analyses — kept, but retargeted:
│   ├── spc.py                    # Rewritten: compute SPC from dataset or from sampled recalls
│   ├── pfr.py                    # Rewritten: both empirical-from-dataset and simulated-from-model paths
│   ├── lag_crp.py                # Rewritten: both empirical-from-dataset and simulated-from-model paths
│   ├── serial_position.py        # DEPRECATED: merged into spc.py (Constitution II)
│   ├── embeddings.py             # Unchanged (unused in 002; retained for future cued-recall work)
│   ├── predicted.py              # Rewritten for v6 retrieval route
│   └── clustering.py             # Unchanged
│
├── figures/                      # 001's figure scripts — rewritten where v1-dependent (FR-048):
│   ├── make_fig_model.py         # Rewritten: v6 block diagram (hierarchical CMR with λ)
│   ├── make_fig_behavioral.py    # New: SPC / pFR / lag-CRP panels (3×2 grid, data vs. model)
│   ├── make_fig_experiment.py    # Unchanged (depicts the 4 empirical conditions)
│   ├── make_fig_analyses.py      # Rewritten if it depended on v1 output
│   ├── make_fig_clustering.py    # Unchanged
│   └── make_param_table.py       # Rewritten: v6 parameter inventory + C&Z Table 1 citations
│
├── notebooks/
│   └── demo.ipynb                # Rewritten: v6 fit + behavioral regression demo; one-cell reproduction of Fig. behavioral
│
├── scripts/
│   ├── compute_embeddings.py     # Unchanged (cued-recall prep, not used in 002)
│   ├── build_reference_curves.py # New: computes empirical SPC/pFR/lag-CRP from data/raw/frfr_category/ and caches under data/processed/reference_curves/ with SHA-256s
│   ├── benchmark_fit.py          # New (FR-031): runs the standard fit command, emits a one-line JSON record
│   ├── check_paper_consistency.py # New (FR-049): greps the compiled paper for v1 artifacts; asserts CornZhan25 in bib
│   ├── reformat_frfr_category.py # Unchanged (FR-061)
│   └── check_no_duplicate_defs.py # Unchanged (Constitution II gate)
│
└── tests/
    ├── test_drift.py              # Replaces test_context.py — v6 item + storyline drift
    ├── test_boundaries.py         # New — event-boundary sync, storyline switch cache, storyline-return reinstatement (v6 Eqs 4-7)
    ├── test_matrices.py           # New — M^IC and M^SC update correctness
    ├── test_preexp.py             # New (US4 / FR-007) — pluggable M^FC_pre
    ├── test_retrieval.py          # New — a = (M^IC)^T c^item, softmax, β_rec drift, ε_d stopping
    ├── test_param_defaults.py     # New (US5 / FR-009) — C&Z Table 1 defaults
    ├── test_hcmr_standard_tcm.py  # λ=0, one storyline → standard CMR equivalence (SC-006)
    ├── test_behavioral_regression.py  # New — Q2 two-layer policy (FR-020 / FR-021 / SC-001)
    ├── test_benchmark.py          # New (FR-031) — asserts Tier-1 wall-clock < 120 s
    ├── test_likelihood.py         # Rewritten for v6 retrieval
    ├── test_fit.py                # Rewritten for v6 parameter set
    ├── test_bootstrap.py          # Rewritten: adds parallelization-determinism assertion
    ├── test_parameter_recovery.py # Rewritten in place on v6 parameters (FR-022)
    ├── test_cross_platform.py     # Rewritten regression fixture on v6 parameters
    ├── test_schema.py             # Unchanged
    ├── test_frfr.py               # Unchanged
    ├── test_dataset.py            # Unchanged
    ├── test_cli.py                # Extended: new subcommands (benchmark, --pre-context, --backend)
    └── test_similarity.py         # DELETED (v1-specific cosine/softmax wrapper; replaced by retrieval.py)

paper/
├── main.tex                      # §3 (Model) and §4 (Derivation) rewritten to v6 (FR-040); new §1.5-equivalent for M^FC_pre (FR-042); v1 §4.4 numerical anchor removed
├── supplement.tex                # Extended: new "v6 vs. v1 migration" section (FR-047, alternative location to notes/v6_migration.md)
├── CDL-bibliography/             # Submodule; read-only in this feature's window
├── local.bib                     # New (FR-043): local bib entry for Cornell & Zhang 2025 if the submodule cannot be edited in this feature
├── figs/
│   ├── fig_model.{pdf,png}       # Regenerated (FR-048)
│   ├── fig_behavioral.{pdf,png}  # New
│   ├── fig_experiment.{pdf,png}  # Unchanged
│   └── source/                   # Regenerated sources
└── compile.sh                    # Unchanged

notes/
├── ms-tcm.pdf                    # Historical; gains a "SUPERSEDED BY v6" watermark header if a watermarking workflow lands, else unchanged
├── two_level_cmr_v6.pdf          # Canonical spec (FR-044)
├── CornZhan25.pdf                # Theoretical predecessor (cited)
└── v6_migration.md               # New (FR-047): v1-symbol → v6-symbol mapping, retirement rationale

specs/
├── 001-ms-tcm-impl/              # Superseded; a note at the top of plan.md points at 002-ms-tcm-v6-hcmr (FR-046)
└── 002-ms-tcm-v6-hcmr/           # This feature (see tree above)

.specify/
└── memory/constitution.md        # MINOR bump 1.0.0 → 1.1.0: canonical spec pointer now notes/two_level_cmr_v6.pdf; §4.4 numerical anchor language replaced

CLAUDE.md                         # Updated (FR-044): v6 canonical spec, v6 parameter regimes, Tier 1 performance claim
README.md                         # Updated (FR-045)
code/README.md                    # Updated (FR-045)
data/README.md                    # Updated (FR-045)
```

**Structure Decision**: **Single Python package** at `code/ms_tcm/` (unchanged from 001). The v6 rewrite replaces module contents and adds new modules for the hierarchical-CMR layer (`hcmr.py`, `drift.py`, `boundaries.py`, `matrices.py`, `preexp.py`, `retrieval.py`) plus Tier-2 JAX as an optional sub-package (`jax_backend/`). The `scripts/` directory gains three new commit-blocking utilities: `benchmark_fit.py` (FR-031 CI gate), `check_paper_consistency.py` (FR-049 paper gate), and `build_reference_curves.py` (FR-020 Layer-2 targets). No new Python package root.

## Complexity Tracking

| Item | Why needed | Simpler alternative rejected because |
|-|-|-|
| Optional JAX backend (`jax_backend/` sub-package) | Tier 2 target of < 30 s requires JIT compilation + autodiff; no Python-level optimization can close the remaining 4× gap between Tier 1's ~120 s and Tier 2's 30 s. | Rewriting the inner loops in Cython or Rust would deliver similar speedups but introduces a compilation step on install, hurting Reproducibility (IV). JAX is pure Python from a packaging perspective and can be opted out of. |
| Multi-dtype CI matrix (float64 / float32) for Tier 2 | Users want both reproducibility and speed; the `MS_TCM_JAX_DTYPE` knob (Q3) exposes the trade-off. CI must verify both paths to prevent a silent divergence. | A single-dtype policy loses either reproducibility or speed and contradicts Q3. |
| `scripts/check_paper_consistency.py` | Documentation rot is the most likely post-merge failure mode: the paper's §3/§4 could drift out of sync with the code during a future feature. A greppable consistency check is the cheapest guard. | Manual review at PR time would miss drift introduced during merge conflicts. |
| `data/processed/reference_curves/` cache | Layer 2 of the Q2 tolerance policy compares simulated curves to empirical curves; computing the empirical curves from scratch in every test run adds 5-10 s per test and pulls pyarrow into a fast-path code path. Pre-computing once and caching with SHA-256 matches the `data/raw/` hash-verified pattern used in 001. | Recomputing every run would slow tests below the 3-minute gate. |
| Constitution minor-bump (1.0.0 → 1.1.0) in the first commit of this feature | The canonical-spec pointer changes from `notes/ms-tcm.pdf` to `notes/two_level_cmr_v6.pdf`; the §4.4 anchor reference is retired. The amendment procedure explicitly allows this as a MINOR. | Leaving the constitution stale would create a Principle-III clarity violation (two competing canonical pointers). |
