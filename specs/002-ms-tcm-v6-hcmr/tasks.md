---
description: "Task list for feature 002-ms-tcm-v6-hcmr: v6 hierarchical-CMR rewrite + fast inference + paper/docs sync"
---

# Tasks: MS-TCM v6 (Hierarchical CMR) Rewrite + Fast Inference + Paper/Docs Sync

**Input**: Design documents from `/specs/002-ms-tcm-v6-hcmr/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/*

**Tests**: MANDATORY. Constitution Principle I requires a hand-derivable unit test for every numerical function; FR-020 pins behavioral regression tests; FR-022 pins parameter recovery. Tests are written FIRST in each phase and must FAIL before implementation lands.

**Organization**: Tasks are grouped by the five user stories from spec.md. MVP = Phase 1 (Setup) + Phase 2 (Foundational) + Phase 3 (US1 behavioral regression). US2, US3, US4, US5 layer additional value on top.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4, US5)
- Include exact file paths in every task

## Path Conventions

- Python package: `code/ms_tcm/` (single source of truth; Constitution II)
- Tests: `code/tests/`
- Scripts: `scripts/`
- Notebooks: `code/notebooks/`
- Figures: `code/figures/` (generators) and `paper/figs/` (output PDFs)
- Paper: `paper/`
- Migration notes: `notes/v6_migration.md`
- Feature spec: `specs/002-ms-tcm-v6-hcmr/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Branch housekeeping, constitution amendment, v1 deletion, dependency bumps. All of these must land before any v6 code is written to avoid mixing old and new.

- [ ] T001 Bump `.specify/memory/constitution.md` from 1.0.0 to 1.1.0 per research.md R0: canonical spec pointer → `notes/two_level_cmr_v6.pdf`; §4.4 anchor language replaced with the `--standard-tcm` reduction anchor; Principle III symbol list updated to v6 (β_enc, β_story, γ_fc, k, λ, β_rec, ε_d, c^item, c^story); update the Sync Impact Report at the file top
- [ ] T002 Create `notes/v6_migration.md` per contracts/paper-consistency.md §8: (a) symbol mapping table (v1 → v6, with the caveat that v1 γ ≠ v6 γ_fc); (b) deletion manifest listing every file/test removed in Phase 1; (c) 3-5 paragraph rationale
- [ ] T003 Update `pyproject.toml`: add `pytest-xdist>=3` to `[project.optional-dependencies].dev`; add new `[project.optional-dependencies].jax = ["jax>=0.4.28", "jaxlib>=0.4.28", "optax>=0.2.0"]`; add new `[project.optional-dependencies].figures = ["matplotlib>=3.8"]`; bump `ms_tcm.__version__` to `"0.2.0"` via the package metadata mechanism already in place
- [ ] T004 [P] Delete v1 modules per FR-013: `code/ms_tcm/composite.py`; `code/ms_tcm/similarity.py`; delete `ms_tcm.mechanisms.apply_resumption_reinstatement`, `apply_conversational_references`, and `interference_factor` leaving the empty module in place for the v6 boundaries.py to eventually take over (but safer: delete `code/ms_tcm/mechanisms.py` entirely and remove it from `__init__.py`)
- [ ] T005 [P] Delete v1 tests: `code/tests/test_similarity.py`, `code/tests/test_composite.py::test_section_4_4_numerical_anchor` and any other tests in `test_composite.py` — then delete `test_composite.py` itself since its only purpose was the §4.4 anchor. Also delete `code/tests/test_context.py` (will be replaced by test_drift.py + test_boundaries.py in Phase 2)
- [ ] T006 [P] Remove v1 fields from `code/ms_tcm/params.py`: `w_global`, `w_storyline`, `w_global_ret`, `w_storyline_ret`, `gamma` (v1 meaning), `alpha_enabled`, `lambda_interference` (v1 meaning), `tau`, `phi_s`, `phi_d`. Leave the file in place as a stub that raises `AttributeError` with a pointer to `notes/v6_migration.md` when any v1 field is accessed; the v6 `ModelParameters` lands in Phase 2
- [ ] T007 Add a SUPERSEDED banner to the top of `specs/001-ms-tcm-impl/plan.md` pointing at `specs/002-ms-tcm-v6-hcmr/plan.md` and `notes/v6_migration.md` per contracts/paper-consistency.md §7
- [ ] T008 Run `pytest code/tests` and confirm it either passes (remaining v1-neutral tests) or produces a clean inventory of v1-specific failures — these are the canaries we are about to replace. Create a git commit with message "v1 retirement: Phase 1 complete" containing T001-T008 changes (A16 resolved: this is a git commit, not a file in `tasks/`).

**Checkpoint**: Constitution amended, v1 code deleted, dependency groundwork in place. No v6 code yet.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core types, entities, and utilities that every user story depends on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T009 Write `code/ms_tcm/params.py` v6 `ModelParameters` dataclass per data-model.md §1: fields β_enc, β_story, γ_fc, k, λ_reinstate, β_rec, ε_d, feature_dim, paradigm, seed; `__post_init__` validation (0 < β < 1, β_enc > β_story, [0,1] weights, k > 0, ε_d > 0); `ModelParameters.standard_tcm()` classmethod that forces λ_reinstate=0; default values cite C&Z 2025 Table 1 and v6 §5 in docstrings (FR-009)
- [ ] T010 [P] Write `code/tests/test_param_defaults.py` (US5 / FR-009): assert default `ModelParameters()` has β_enc=0.679, β_story=0.400, γ_fc=0.315, k=6.50, λ_reinstate=0.80, β_rec=0.326, ε_d=1.04 to exact equality; assert each field's docstring contains the substring "Cornell & Zhang 2025 Table 1" or "v6 §5"
- [ ] T011 [P] Write `code/ms_tcm/preexp.py` per data-model.md §2: `MFCPreMatrix` Protocol with `n_features`, `n_items`, `apply(item_indices) -> ndarray`; concrete `IdentityPreMatrix(n_items, n_features)` (returns one-hot rows); concrete `EmbeddingPreMatrix(embeddings: ndarray, item_to_index: dict)` with a `from_parquet(path)` classmethod (the Parquet loader is a stub that raises NotImplementedError — this feature does NOT implement USE integration per spec §Non-goals)
- [ ] T012 [P] Write `code/tests/test_preexp.py` (US4 / FR-007): assert `IdentityPreMatrix(15, 71).apply(np.array([3]))` returns shape (1,71) with a single 1.0 at column 3; assert `EmbeddingPreMatrix` with a hand-constructed dummy embeddings matrix returns the matched row; assert `from_parquet` on a real path raises NotImplementedError with a clear message
- [ ] T013 Implement `code/ms_tcm/matrices.py` per data-model.md §5: `ic_update(M_IC, c_item, f_i) -> None` (in-place); `sc_update(M_SC, storyline_onehot, c_story_out) -> None`; `fc_exp_update(M_FC_exp, c_in, f_i) -> None`; `read_cached_story(M_SC, storyline_onehot) -> ndarray`. All four operate on preallocated arrays in-place (Tier 1 perf hygiene).
- [ ] T014 [P] Write `code/tests/test_matrices.py`: hand-derive one-step M^IC from a 3-dim c_item and 2-item f; assert `ic_update` matches to 1e-12; analogous tests for `sc_update` and `fc_exp_update`; assert `read_cached_story` reads back the exact cached vector at 1e-12
- [ ] T015 [P] Vectorize `code/ms_tcm/features.py::encode_features` per FR-030 Tier 1 optimization: replace `iterrows()` with vectorized numpy operations using `df.to_numpy()` columnwise; preserve exact bit-for-bit output for FRFR-category input; add a benchmark assertion in `test_features.py::test_vectorization_bit_identity` comparing against the old iterrows output saved as a golden file
- [ ] T016 [P] Vectorize `code/ms_tcm/features.py`: add a `DatasetFeatureCache` class that holds the feature matrix and M^FC_pre, keyed by dataset `id()`, exposed as `get_or_compute(dataset, pre_matrix)` (FR-030 cache optimization). Do NOT wire it into `likelihood.py` yet — that happens in Phase 6

**Checkpoint**: Foundational v6 types ready. ModelParameters + MFCPreMatrix + matrices + vectorized features exist with tests.

---

## Phase 3: User Story 1 — Behavioral regression (Priority: P1) 🎯 MVP

**Goal**: MS-TCM reproduces SPC, pFR, and lag-CRP on FRFR-category per the Q2 two-layer policy.

**Independent Test**: `pytest code/tests/test_behavioral_regression.py -v` passes both Layer 1 (qualitative shape) and Layer 2 (≤ 20 % relative error vs. FRFR-category empirical curves, MS-TCM no-worse-than `--standard-tcm`).

### Tests for User Story 1 (write first; they must fail before implementation lands)

- [ ] T017 [P] [US1] Write `code/tests/test_drift.py` per v6 §2.1: `test_item_drift_preserves_unit_norm`, `test_story_drift_preserves_unit_norm`, `test_item_drift_with_beta_enc_0p679_matches_hand_derivation`, `test_story_drift_with_beta_story_0p400_matches_hand_derivation` (1e-12 tolerance)
- [ ] T018 [P] [US1] Write `code/tests/test_boundaries.py` per v6 §2.2-2.4: `test_event_boundary_syncs_item_to_story`, `test_storyline_switch_caches_to_MSC_and_syncs_item`, `test_storyline_return_reinstates_with_lambda_blend`, `test_storyline_return_with_lambda_zero_equals_normal_switch`, `test_storyline_return_with_lambda_one_fully_restores_cached_context`
- [ ] T019 [P] [US1] Write `code/tests/test_retrieval.py` per FR-005, FR-006: `test_activation_equals_MIC_T_times_c_item_cue`, `test_softmax_with_gain_k`, `test_beta_rec_drifts_retrieval_toward_last_recalled`, `test_epsilon_d_stopping_rule`
- [ ] T019b [P] [US1] Write `code/tests/test_paradigm.py` (covering FR-006 cued-recall path, A7): `test_cued_recall_paradigm_ignores_beta_rec_and_epsilon_d` — construct a `ModelParameters(paradigm="cued_recall")` with deliberately extreme β_rec and ε_d, call `score_cue(...)`, assert the returned probabilities do NOT depend on β_rec (no retrieval drift between cues) and do NOT invoke ε_d stopping; compare to the same probabilities from a model built with default β_rec/ε_d, assert bit-identity
- [ ] T020 [P] [US1] Write `code/tests/test_hcmr_standard_tcm.py` per SC-006: construct a 1-storyline 3-word list, run HierarchicalCMRModel with λ=0, assert the resulting M^IC and retrieval probabilities match a hand-derived standard-CMR computation at 1e-10
- [ ] T021 [US1] Write `scripts/build_reference_curves.py` per FR-020 Layer 2: reads `data/raw/frfr_category/`, computes empirical SPC/pFR/lag-CRP via `code/analyses/{spc,pfr,lag_crp}.py`, writes Parquet files to `data/processed/reference_curves/` + `manifest.json` with SHA-256s; idempotent (same input → byte-identical output)
- [ ] T022 [US1] Write `code/tests/test_behavioral_regression.py` per contracts/regression-tests.md: Layer 1 (3-participant smoke fit; 6 qualitative assertions cited to C&Z 2025 Fig 2); Layer 2 (50-bootstrap medium fit on full dataset; per-bin relative error ≤ 20 % + MS-TCM no-worse-than standard-TCM on each of SPC/pFR/lag-CRP); second parameterization that uses `--standard-tcm` as the primary model and skips Gate 2

### Implementation for User Story 1

- [ ] T023 [US1] Implement `code/ms_tcm/drift.py` per data-model.md §3: `update_item_context(c_prev, beta_enc, c_in) -> c_new` and `update_story_context(c_prev, beta_story, c_in) -> c_new`; each computes ρ = √(1-β²) internally (FR-008); freeze semantics handled by the caller (orchestrator in hcmr.py)
- [ ] T024 [US1] Implement `code/ms_tcm/boundaries.py` per data-model.md §4: `apply_event_boundary(c_item, c_story) -> c_item_new` (synchronize); `apply_storyline_switch(M_SC, outgoing_storyline_idx, c_story_out, c_story_new) -> (updated_M_SC, c_item_new)`; `apply_storyline_return(M_SC, returning_storyline_idx, c_story_prev, lambda_reinstate) -> c_story_new`
- [ ] T025 [US1] Implement `code/ms_tcm/retrieval.py` per FR-005, FR-006: `activation(M_IC, c_item_cue) -> ndarray` (= M_IC.T @ c_item_cue); `recall_probabilities(activation, k, mask=None) -> ndarray` (softmax with gain k, optional mask for already-recalled); `drift_retrieval_context(c_ret, beta_rec, c_item_recalled) -> c_ret_new`; `stopping_probability(a_r_sum, a_nr_sum, epsilon_d) -> float` (per C&Z 2025 Eq 7)
- [ ] T026 [US1] Implement `code/ms_tcm/hcmr.py::HierarchicalCMRModel.encode` per data-model.md §6: iterate (participant, list), reset c_item/c_story to e_start at list boundary; per step determine (event_boundary, storyline_switch, storyline_return) flags from e(i)/s(i); route to the correct boundary function from T024; drift the active storyline via T023; update M^IC, M^SC, M^FC_exp via T013; record `c_item`, `c_story`, `c_composite`, `M_IC`, `M_SC`, `active_storyline` in an EncodingState frozen dataclass
- [ ] T027 [US1] Implement `code/ms_tcm/hcmr.py::HierarchicalCMRModel.score_first_recall` / `score_next_recall` / `score_cue` per data-model.md §6, using T025 for activation and probabilities; `score_next_recall` implements the β_rec retrieval drift (FR-006)
- [ ] T028 [US1] Implement `code/ms_tcm/hcmr.py::sample_recalls` per FR-012: deterministically samples a `recalled.parquet`-shaped pyarrow Table from `score_first_recall` / `score_next_recall`, using the ε_d stopping rule to determine recall length per list; accepts an optional `recall_length_fn` override for tests
- [ ] T029 [US1] Rewrite `code/analyses/spc.py`, `code/analyses/pfr.py`, `code/analyses/lag_crp.py` per contracts/model-api.md §7: each exposes a `compute_<curve>(recalled_table_or_dataset) -> pd.Series` that dispatches on input type (Dataset = empirical, pa.Table = simulated). Delete `code/analyses/serial_position.py` (merged into `spc.py`, Constitution II)
- [ ] T030 [US1] Rewrite `code/ms_tcm/likelihood.py` per the v6 retrieval route: `list_log_likelihood(presented, recalled, parameters, state, participant, list_) -> float` — iterate observed recall transitions, evaluate softmax probability via the new `retrieval.py`, include ε_d stopping-rule contribution for free-recall paradigm, sum log-probabilities; intrusions (sp==0) skipped; `dataset_log_likelihood` unchanged in signature
- [ ] T031 [US1] Rewrite `code/ms_tcm/fit.py` reparameterization per contracts/fitter.md §3: 7 unconstrained parameters (6 for cued-recall); β_story/β_enc ratio as a logit; C&Z 2025 Table 1 defaults as starting point for random restarts; L-BFGS-B with finite-difference jacobian (Tier 1)
- [ ] T032 [US1] Rewrite `code/ms_tcm/bootstrap.py` with `multiprocessing.Pool` parallelism per FR-030: workers receive numpy-array work items (not pyarrow Dataset); deterministic seed derivation per bootstrap draw; `test_bootstrap.py::test_parallel_matches_serial_same_seed` verifies determinism
- [ ] T033 [US1] Run the behavioral regression test (T022) end-to-end; iterate on T023-T032 until both layers pass. No mocks, no shortcuts — if Layer 2 fails, the root cause is in the model/likelihood code, not the test

**Checkpoint**: MS-TCM reproduces SPC/pFR/lag-CRP on FRFR-category at the Q2 bar. This is the MVP.

---

## Phase 4: User Story 2 — Fast fit wall-clock (Priority: P1)

**Goal**: `ms-tcm fit` completes in < 120 s (Tier 1) and optionally < 30 s (Tier 2 JAX).

**Independent Test**: `python scripts/benchmark_fit.py data/raw/frfr_category --tier tier1 --seed 42 --n-bootstraps 1000 --n-restarts 5` returns wall-clock < 120 s on CI hardware.

### Tests for User Story 2 (write first)

- [ ] T034 [P] [US2] Write `code/tests/test_benchmark.py::test_tier1_wall_clock_under_120s` per FR-031: invokes `benchmark_fit` with default settings, asserts wall_clock_seconds < 120.0. Marked `@pytest.mark.slow`; run on CI but not every local iteration
- [ ] T035 [P] [US2] Write `code/tests/test_bootstrap.py::test_parallel_matches_serial_same_seed`: same dataset + same seed, run bootstrap serial and parallel, assert `bootstrap_draws` are bit-identical
- [ ] T036 [P] [US2] Write `code/tests/test_cross_platform.py::test_tier1_mle_within_1e_minus_10`: hand-seeded small dataset, run `fit_mle` twice, assert MLE vectors agree to 1e-10 (FR-023)

### Implementation for User Story 2 (Tier 1 — mandatory)

- [ ] T037 [US2] Vectorize the per-list encoding recurrence in `code/ms_tcm/hcmr.py::encode` per research.md R2.3: for constant β, express each storyline's context trajectory as a matrix operation over (W+1, d) and avoid the inner Python `for t in range(W)` loop. Verify via `test_drift.py` that outputs are bit-identical to the scalar-loop version
- [ ] T038 [US2] Wire `DatasetFeatureCache` (from T016) into `code/ms_tcm/likelihood.py::dataset_log_likelihood` and `bootstrap.py::_resample_participants` so the feature matrix + M^FC_pre are computed once per Dataset and reused across all likelihood evaluations in a fit (FR-030 cache optimization)
- [ ] T039 [US2] Write `code/ms_tcm/benchmark.py::benchmark_fit` per FR-031 and contracts/model-api.md §6: runs a deterministic fit, measures wall-clock via `time.perf_counter()` and peak memory via `resource.getrusage(RUSAGE_SELF).ru_maxrss`, returns a dict with every field in data-model.md §8
- [ ] T040 [US2] Write `scripts/benchmark_fit.py` CLI wrapper per contracts/cli.md `ms-tcm benchmark`: reads args, calls `benchmark_fit`, appends one row to `data/processed/benchmarks/benchmark_log.csv`, prints JSON to stdout, exits non-zero if wall-clock exceeds tier gate
- [ ] T041 [US2] Add `ms-tcm benchmark` subcommand to `code/ms_tcm/cli.py` matching contracts/cli.md; route `--tier tier1|jax` to the appropriate backend
- [ ] T042 [US2] Add a CI job to `.github/workflows/ci.yml` that runs `python scripts/benchmark_fit.py data/raw/frfr_category --tier tier1 --seed 42 --n-bootstraps 1000 --n-restarts 5` and asserts exit code 0. On Ubuntu runner (most stable CPU timings); the macOS and Windows runners get the same invocation as a sanity check but without the hard wall-clock gate

### Implementation for User Story 2 (Tier 2 — optional, gated by feasibility)

- [ ] T043 [P] [US2] Write `code/ms_tcm/jax_backend/hcmr_jax.py` per FR-032: JAX-native re-implementation of `encode` + `score_*` using `jax.numpy` and `jax.lax.scan` for the encoding recurrence; `jax.jit`-compiled; respects `jax_enable_x64` config toggle for dtype
- [ ] T044 [P] [US2] Write `code/ms_tcm/jax_backend/fit_jax.py` per FR-032: Optax L-BFGS with `jax.grad(negative_log_likelihood)` as the objective; API-identical to `scipy.optimize.minimize` wrapper so `fit.py` can swap backends without public changes
- [ ] T045 [US2] Wire `MS_TCM_BACKEND=jax` / `--backend jax` switch in `code/ms_tcm/cli.py` and `code/ms_tcm/fit.py`; honor `MS_TCM_JAX_DTYPE=float32|float64` via `jax.config.update("jax_enable_x64", ...)` at import time (FR-032, Q3); implement the JAX-not-installed auto-fallback per spec Assumptions (A6 resolved): if the JAX backend is requested but `jax` is not importable, log a warning and fall back to Tier 1 rather than failing
- [ ] T045b [US2] Write `code/tests/test_cli.py::test_jax_fallback_when_unavailable` (A6 resolved): monkey-patch `jax` import to raise ImportError; invoke the CLI with `--backend jax`; assert (a) a warning with substring "falling back to Tier 1" is emitted via the `warnings` module or stderr log, (b) the fit completes successfully, (c) exit code is 0 (not 3)
- [ ] T046 [P] [US2] Write `code/tests/test_jax_backend.py` — skipped when `jax` not installed (`@pytest.mark.skipif(...)`); assert JAX MLE matches Tier 1 MLE within 1e-10 (float64) or 1e-8 (float32)
- [ ] T047 [US2] Write `code/tests/test_benchmark.py::test_tier2_wall_clock_under_30s` — skipped when `jax` not installed; assert Tier 2 wall-clock < 30 s
- [ ] T048 [US2] Extend `.github/workflows/ci.yml` with a `jax`-extras install job and the `MS_TCM_JAX_DTYPE=float64|float32` matrix dimension

**Checkpoint**: Tier 1 fit under 120 s on CI; Tier 2 optional, under 30 s when installed.

---

## Phase 5: User Story 3 — Paper and documentation match v6 (Priority: P1)

**Goal**: `paper/main.tex`, `CLAUDE.md`, and the three READMEs describe the v6 model using v6 notation and cite Cornell & Zhang 2025.

**Independent Test**: `python scripts/check_paper_consistency.py` exits 0; a human reviewer reads `paper/main.pdf` and finds the v6 hierarchical CMR formulation in §3, a CornellZhang2025 citation, and no references to v1 artifacts.

### Tests for User Story 3 (write first)

- [ ] T049 [P] [US3] Write `scripts/check_paper_consistency.py` per contracts/paper-consistency.md §9: greps `paper/main.tex` body (stripping comments + the migration note if it's in supplement.tex) for each banned string from §2.4; asserts each required string from §2.5 is present; verifies CornellZhang2025 entry in bib. Exit codes per §9
- [ ] T050 [P] [US3] Write `code/tests/test_paper_consistency.py` that shells out to `scripts/check_paper_consistency.py` and asserts exit code 0 after User Story 3 implementation lands; marked `@pytest.mark.slow` because it runs `paper/compile.sh` first to ensure the checked file is actually built

### Implementation for User Story 3

- [ ] T051 [US3] Rewrite `paper/main.tex` §3 (Model) per contracts/paper-consistency.md §2.1 to match v6 §1-3: hierarchical CMR with item-level + storyline-level context, boundary sync, M^IC / M^SC, λ storyline-return reinstatement; include the sentence "MS-TCM adds exactly one mechanism to Cornell & Zhang's (2025) hierarchical CMR: storyline-return reinstatement at encoding-time with strength λ"; cite CornellZhang2025 in this section
- [ ] T052 [US3] Rewrite `paper/main.tex` §4 (Derivation) per contracts/paper-consistency.md §2.1: derive the grouped-vs-bridge equivalence via v6 mechanisms, not v1 reweighting; delete §4.4 numerical anchor entirely
- [ ] T053 [US3] Add a new §1.5-equivalent subsection to `paper/main.tex` per FR-042: M^FC_pre identity-vs-embeddings discussion from v6 §1.5; cite Polyn et al. 2009, Morton & Polyn 2016, Heusser et al. 2021, Xu et al. 2024 as the embedding-precedent lineage (v6 §1.5.1)
- [ ] T054 [US3] Update `paper/main.tex` §Methods parameter table per contracts/paper-consistency.md §2.3: v6 symbols with C&Z 2025 Table 1 starting values and v6 §5 starting value for λ; remove any v1 parameters
- [ ] T055 [P] [US3] Create `paper/local.bib` per FR-043 with the `@article{CornellZhang2025, ...}` entry (author, title, journal = Psychological Review, year = 2025, with DOI or preprint URL per research.md R5)
- [ ] T055b [P] [US3] Extend `scripts/check_paper_consistency.py` (T049) with an SC-008 sub-check (A3): parse the resolved bib entry for `CornellZhang2025` and assert (a) `author = {Cornell, Charlotte A. and Zhang, Qiong}` (or equivalent with middle initials), (b) `journal = {Psychological Review}`, (c) `year = {2025}`, (d) one of `doi` OR `url` fields is present and non-empty. Exit code 3 (per contracts/paper-consistency.md §9) on any missing field. A CI job asserts the bib entry resolves
- [ ] T056 [US3] Update `paper/main.tex` `\bibliography{}` command to include both `CDL-bibliography/cdl` and `local` if `local.bib` is the chosen target; otherwise add the entry to the CDL submodule if editable and skip `local.bib`
- [ ] T057 [US3] Write `code/figures/make_fig_model.py` per FR-048: regenerate the Figure 1 concept diagram for the v6 hierarchical CMR; save as `paper/figs/fig_model.pdf` at low DPI (≤ 100) for Claude review per global image-resolution guidance
- [ ] T058 [US3] Write `code/figures/make_fig_behavioral.py` per FR-048: new figure with three panels (SPC / pFR / lag-CRP), each showing FRFR-category empirical data alongside MS-TCM and standard-TCM simulated curves; save to `paper/figs/fig_behavioral.pdf`. Notebook at `code/notebooks/demo.ipynb` reproduces this figure
- [ ] T059 [P] [US3] Update `CLAUDE.md` per contracts/paper-consistency.md §5: canonical spec pointer → `notes/two_level_cmr_v6.pdf`; parameter-regimes list → v6 symbols; §4.4 paragraph replaced with `test_hcmr_standard_tcm.py` + behavioral regression anchor; add one-liner for `ms-tcm benchmark` + the Tier 1 target
- [ ] T060 [P] [US3] Update `README.md` per contracts/paper-consistency.md §6: replace CDL-template "Paper title" placeholder with project-specific text; add Tier 1 performance claim; point at `specs/002-ms-tcm-v6-hcmr/` as active feature
- [ ] T061 [P] [US3] Update `code/README.md`: CLI examples with v6 parameter names; add `benchmark` subcommand; pointer to `specs/002-ms-tcm-v6-hcmr/contracts/cli.md`
- [ ] T062 [P] [US3] Update `data/README.md`: note new `data/processed/reference_curves/` directory and its regeneration command; FRFR-category docs unchanged
- [ ] T063 [US3] Rebuild the paper (`cd paper && ./compile.sh`) and verify `main.pdf` + `supplement.pdf` compile without warnings; commit the rebuilt PDFs per CDL convention

**Checkpoint**: Paper, CLAUDE.md, and READMEs describe v6 accurately. CornellZhang2025 is cited. `check_paper_consistency.py` exits 0.

---

## Phase 6: User Story 4 — Pluggable M^FC_pre (Priority: P2)

**Goal**: A researcher can hand a pre-computed embedding matrix to the model via the `MFCPreMatrix` API and `--pre-context` CLI flag, without touching any other code.

**Independent Test**: `pytest code/tests/test_preexp.py` (from T012 in Phase 2) plus a new integration test that uses a dummy embeddings matrix and asserts the item-context trajectories differ from identity.

### Tests for User Story 4

- [ ] T064 [P] [US4] Extend `code/tests/test_preexp.py` with `test_embeddings_shift_item_context_trajectories`: construct a small dummy embeddings matrix that makes items 0 and 1 highly similar; run `HierarchicalCMRModel.encode` on a 2-item list; assert `c_item` trajectory differs measurably from the identity-matrix case
- [ ] T065 [P] [US4] Write `code/tests/test_cli.py::test_pre_context_flag_accepts_identity_and_parquet_path`: assert CLI accepts `--pre-context identity` and `--pre-context <path>.parquet`; for a non-existent path, asserts CLI exits with code 2

### Implementation for User Story 4

- [ ] T066 [US4] Thread the `MFCPreMatrix` object through `HierarchicalCMRModel.__init__` per data-model.md §2; defaults to `IdentityPreMatrix(feature_dim, feature_dim)` when not supplied
- [ ] T067 [US4] Implement M^FC_pre application in `code/ms_tcm/hcmr.py::encode`: before drift at each step, compute `c^IN_i = (1 - γ_fc) · M^FC_pre.apply(i) + γ_fc · (M^FC_exp @ f_i)` per v6 Eq 1.5.1
- [ ] T068 [US4] Add `--pre-context identity|<path>` flag to `code/ms_tcm/cli.py`; for `identity` use `IdentityPreMatrix`; for a path, use `EmbeddingPreMatrix.from_parquet(path)` (which still raises NotImplementedError per T011 since actual USE loading is out of scope)

**Checkpoint**: The model is "cued-recall ready" — swapping in a pre-experimental embeddings matrix works at the API level.

---

## Phase 7: User Story 5 — Parameter starting values cite C&Z Table 1 (Priority: P2)

**Goal**: `code/ms_tcm/params.py` and the paper Methods section both explicitly cite Cornell & Zhang 2025 Table 1 (and v6 §5) as the source of starting values.

**Independent Test**: `pytest code/tests/test_param_defaults.py` (T010 already covers the code side); a human reviewer opens `paper/main.pdf` §Methods and sees the citations.

### Tests for User Story 5

- [ ] T069 [US5] Extend `code/tests/test_param_defaults.py` with `test_param_docstrings_cite_source`: parse the `ModelParameters` dataclass field docstrings (via `inspect` or `dataclasses.fields`) and assert each one mentions either "Cornell & Zhang 2025 Table 1" or "v6 §5"

### Implementation for User Story 5

- [ ] T070 [US5] (Already satisfied by T009 and T054; this task verifies them.) Re-read `code/ms_tcm/params.py` and `paper/main.tex` §Methods; ensure every parameter's starting value has an in-line citation. Fix any drift.

**Checkpoint**: Parameter provenance is traceable from code to C&Z 2025.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final hygiene, parameter-recovery validation, cross-platform verification, and commit of all derived artifacts.

- [ ] T071 [P] Run `python scripts/check_no_duplicate_defs.py` and confirm zero duplicate function definitions across `code/ms_tcm/` + `code/notebooks/` (Constitution II / FR-062)
- [ ] T072 [P] Rewrite `code/tests/test_parameter_recovery.py` per FR-022: synthesize recalls from v6 MS-TCM with a known θ_true; fit; assert θ_true lies inside 95 % bootstrap CI for ≥ 95 % of parameters. Marked `@slow`; runs on CI once per nightly
- [ ] T073 [P] Run `pytest code/tests` on macOS, Ubuntu (via CI); assert all green. The cross-platform `test_cross_platform.py::EXPECTED_MLE` regression fixture is updated in-place with the new v6 parameter values
- [ ] T074 [P] Rebuild `paper/compile.sh` and commit `main.pdf` + `supplement.pdf` (CDL convention: the committed PDFs must match what the code reproduces)
- [ ] T075 [P] Run `python scripts/benchmark_fit.py data/raw/frfr_category --tier tier1 --seed 42 --n-bootstraps 1000 --n-restarts 5` on a local laptop; record the wall-clock in `data/processed/benchmarks/benchmark_log.csv` and confirm it's below 120 s even with the CI overhead factored out
- [ ] T076 Run `python scripts/check_paper_consistency.py` and confirm exit code 0
- [ ] T077 [P] Run `ms-tcm validate data/raw/frfr_category` and confirm exit code 0 (FR-061: dataset still byte-identical)
- [ ] T078 Run the full quickstart.md end-to-end on a fresh clone (or a container mimicking one) — validate every step in order; any failure is a documentation bug to fix
- [ ] T078b [P] (A4 resolved) Write `scripts/check_figure_provenance.py`: enumerate every `paper/figs/*.pdf` referenced by `\includegraphics{}` in `paper/main.tex`, and assert each has a corresponding generator under `code/figures/` or a notebook cell under `code/notebooks/`. Flag any figure without a regeneration path as an orphan (exit non-zero). Run from Phase-8 polish and wire into CI
- [ ] T078c [P] (A15 resolved) Create `specs/002-ms-tcm-v6-hcmr/checklists/documentation-review.md` with explicit checkboxes: `[ ] CLAUDE.md updated to v6`, `[ ] README.md placeholder replaced`, `[ ] code/README.md v6 parameter names`, `[ ] data/README.md reference_curves section`, `[ ] specs/001-ms-tcm-impl/plan.md has SUPERSEDED banner`, `[ ] specs/001-ms-tcm-impl/contracts/ files either updated or forward-pointed`, `[ ] notes/v6_migration.md has all three required sections`, `[ ] paper/main.tex §3 v6`, `[ ] paper/main.tex §4 v6`, `[ ] paper/main.tex Methods table v6 parameters`, `[ ] paper/main.tex cites CornZhan25`, `[ ] CornZhan25 entry present in bib with DOI/URL`. Reviewer fills this checklist before merging. The file is committed per FR-046
- [ ] T078d (A12 resolved) Extend `scripts/check_paper_consistency.py` (T049) with a migration-doc section check: assert `notes/v6_migration.md` exists AND contains substring "symbol mapping" (case-insensitive) AND "deletion manifest" (case-insensitive) AND "rationale" (case-insensitive); exit code 4 on failure. Equivalent grep: `grep -iE "symbol mapping|deletion manifest|rationale" notes/v6_migration.md` must return at least 3 distinct matches
- [ ] T079 Open the pull request targeting `main` (after 001 merges or rebasing onto 001); include a summary referencing spec.md, plan.md, and the three clarifications (Q1/Q2/Q3); paste the completed `specs/002-ms-tcm-v6-hcmr/checklists/documentation-review.md` into the PR body

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately. **BLOCKS** everything downstream via the constitution bump.
- **Foundational (Phase 2)**: Depends on Setup. **BLOCKS** all user stories.
- **US1 (Phase 3)**: Depends on Foundational. This is the MVP — nothing else can land until US1 proves the model is correct.
- **US2 (Phase 4)**: Depends on US1 (needs a working fit to benchmark). Tier 1 (T037-T042) mandatory; Tier 2 (T043-T048) optional.
- **US3 (Phase 5)**: Depends on US1 (needs a working model to describe in the paper) + Phase 2. Paper edits can happen in parallel with US2 performance work.
- **US4 (Phase 6)**: Depends on US1. Independent of US2/US3 at the code level.
- **US5 (Phase 7)**: Depends on US1 (paper citations) + US3 (paper itself exists). Partially redundant with T009/T054 — T069/T070 are verification.
- **Polish (Phase 8)**: Depends on all user stories.

### User Story Dependencies

- **US1 (P1)**: After Foundational. This is the BLOCKING MVP — without a correct model, benchmarking, paper edits, and the pre-context hook are all meaningless.
- **US2 (P1)**: After US1. Can be parallelized with US3.
- **US3 (P1)**: After US1. Can be parallelized with US2.
- **US4 (P2)**: After US1.
- **US5 (P2)**: After US1 + US3.

### Within Each User Story

- Tests MUST be written before implementation (Constitution I + Phase-level TDD).
- `params.py` / `preexp.py` / `matrices.py` (Foundational) before anything that consumes them.
- `drift.py` → `boundaries.py` → `matrices.py` → `retrieval.py` → `hcmr.py` (US1 order).
- `likelihood.py` → `fit.py` → `bootstrap.py` (US1 tail).
- Paper body → bib → figures → compile (US3 order).

### Parallel Opportunities

- Setup tasks T004, T005, T006 can run in parallel.
- Foundational tasks T010, T011, T012, T014, T015, T016 can run in parallel (after T009).
- US1 tests T017, T018, T019, T020 can run in parallel.
- US1 implementation: T023-T025 must be sequential; T029 (analyses) can run in parallel with T023-T025.
- US2 Tier-1 tests T034, T035, T036 can run in parallel.
- US2 Tier-2 tasks T043, T044, T046 can run in parallel (with US2 Tier-1 already complete).
- US3 doc updates T059, T060, T061, T062 can run in parallel (after T051-T058 land).
- US4 tests T064, T065 can run in parallel.
- Polish tasks T071, T072, T073, T074, T075, T077 can run in parallel.

---

## Parallel Example: US1 tests

```bash
# All four US1 test files can be authored in parallel:
Task: "Write code/tests/test_drift.py per v6 §2.1"
Task: "Write code/tests/test_boundaries.py per v6 §2.2-2.4"
Task: "Write code/tests/test_retrieval.py per FR-005, FR-006"
Task: "Write code/tests/test_hcmr_standard_tcm.py per SC-006"
```

---

## Implementation Strategy

### MVP first (Setup + Foundational + US1)

1. Complete Phase 1: constitution, v1 deletion, deps.
2. Complete Phase 2: ModelParameters, MFCPreMatrix, matrices, vectorized features.
3. Complete Phase 3 (US1): model correctness — SPC/pFR/lag-CRP regression green.
4. **STOP and VALIDATE**: the model is correct. Everything downstream is either speed (US2) or documentation (US3/US4/US5).

### Incremental delivery

- MVP: Setup + Foundational + US1 → model reproduces free-recall phenomena. (Shippable as a preprint-quality result.)
- +US2 Tier 1: fit under 2 minutes. (Unblocks rapid iteration for the paper.)
- +US3: paper and docs sync. (Ready for internal review.)
- +US4: pre-context hook. (Sets up cued-recall fit in a future feature.)
- +US5: param citation audit. (Pre-submission polish.)
- +US2 Tier 2 (optional): JAX path. (Nice-to-have.)
- +Polish: parameter-recovery + CI green everywhere. (Ready for PR.)

### Parallel team strategy

After Phase 3 (US1) lands:

- Developer A: Phase 4 (US2 Tier 1 perf), then Phase 4 Tier 2 if capacity.
- Developer B: Phase 5 (US3 paper/docs) in parallel with Developer A.
- Developer C: Phase 6 (US4 pre-context) + Phase 7 (US5 citations) + kick off Phase 8 polish.

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Tests MUST be written and verified to fail before implementation code lands (Constitution Principle I)
- Commit after each task or logical group; outstanding changes committed before any in-place rewrite (Constitution II)
- Every numerical assertion MUST cite a section of `notes/two_level_cmr_v6.pdf`, a `notes/CornZhan25.pdf` equation/table, the FRFR-category `manifest.json`, or a Manning et al. (2023) reference (Constitution III + SC-005)
- Stop at any checkpoint to validate the story independently
- No mocks; no stubs; no silent fallbacks (global `CLAUDE.md` rule)
- The Xu et al. 2026 cued-recall fit is NOT part of this feature (spec §Non-goals) — the pre-context hook (US4) only prepares for it
- If Tier 2 (JAX) cannot land within this feature's effort budget, T043-T048 become a follow-on feature; Tier 1 (T037-T042) is non-negotiable
