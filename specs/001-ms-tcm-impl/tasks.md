---
description: "Task list for feature 001-ms-tcm-impl: MS-TCM model + FRFR-category worked example + MLE/CI fitter"
---

# Tasks: MS-TCM Model Implementation with FRFR-Category Worked Example

**Input**: Design documents from `/specs/001-ms-tcm-impl/` (spec.md, plan.md; research.md and contracts/* are Phase 1 artifacts that will be updated alongside this task list as implementation lands)
**Prerequisites**: plan.md (tech stack, structure, constitution check), spec.md (user stories + 2026-04-20 and 2026-04-21 Clarifications)

**Tests**: INCLUDED. FR-013 requires named unit tests; SC-002/SC-005/SC-009 pin numerical assertions. Tests are written **first** for each story so they fail before implementation lands.

**Organization**: Tasks grouped by user story from `spec.md`. MVP = Setup + Foundational + US1 + US2 + US3 (model running on the bundled FRFR-category dataset). US4 (MLE/CI fitter) is the scientific endpoint of this feature.

## Path Conventions

- Python package: `code/ms_tcm/` (single source of truth; Constitution II)
- Tests: `code/tests/`
- Notebooks: `code/notebooks/`
- Reformat script: `scripts/reformat_frfr_category.py` (already committed)
- Bundled dataset: `data/raw/frfr_category/` (already committed; Parquet + CSV + manifest)
- Packaging: `pyproject.toml` at repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization that all user stories depend on

- [X] T001 Create `pyproject.toml` at repository root declaring the `ms_tcm` package (located at `code/ms_tcm/`), Python 3.11 minimum, dependencies `numpy>=1.26`, `scipy>=1.11`, `pyarrow>=14`, `pandas>=2.1`, `h5py>=3.10`, `pytest>=7`, `pytest-cov`, and `[project.scripts]` entry for `ms-tcm = "ms_tcm.cli:main"` per contracts/cli.md
- [X] T002 Create package skeleton at `code/ms_tcm/__init__.py` re-exporting the public API named in contracts/model-api.md (each name implemented in its own module below)
- [X] T003 [P] Configure `pytest` discovery in `pyproject.toml` (testpaths = `code/tests`, strict markers, a shared `atol=1e-12` fixture for numerical assertions)
- [X] T004 [P] Update `Dockerfile` at repository root: replace `contextlab/cdl-python:3.7` base with `python:3.11-slim`; remove template `brainiak`/`hypertools`/`pandas=1.0.5` pins; add `pip install -e ".[dev]"` as the build step; keep `WORKDIR /mnt` and the jupyter-friendly entrypoint (plan.md Complexity Tracking row 1)
- [X] T005 [P] Add `.gitignore` entries for `data/processed/fits/`, `data/raw/frfr_category/exp2.egg.cache`, `*.egg-info/`, `.pytest_cache/`, `__pycache__/`; verify the last three are not duplicated

**Checkpoint**: `pip install -e ".[dev]"` succeeds on macOS, Ubuntu, and Windows; `pytest code/tests` runs (zero tests collected is fine) before any user story begins

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core types and utilities that every user story depends on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T006 Implement `ModelParameters` frozen dataclass in `code/ms_tcm/params.py` per contracts/model-api.md — fields (β_G, β_S, w_G, w_S, w_G^ret, w_S^ret, γ, α_enabled, λ, feature_dim, seed), `__post_init__` validation (sum-to-1 within 1e-12, β bounds, non-negative γ/λ), and `ModelParameters.standard_tcm()` classmethod that returns w_S = 0 and disables storyline updates
- [X] T007 Implement SHA-256 hashing and deterministic JSON writer in `code/ms_tcm/io.py` — `sha256_file(path)`, `write_json_sorted(obj, path)` (two-space indent, sorted keys, newline-terminated)
- [X] T008 [P] Define Apache Arrow schemas and manifest contract in `code/ms_tcm/schema.py` per contracts/dataset-schema.md — `PRESENTED_SCHEMA`, `RECALLED_SCHEMA`, a `MANIFEST_KEYS` constant enumerating the top-level keys required by FR-009 (`source`, `design`, `files`, `row_counts`, `created_at`) plus the required sub-keys for each, and a `ValidationRule` enum covering the seven FR-012 classes (implementations land in US2). Resolves C2.
- [X] T009 [P] Implement numerically-guarded `cosine_similarity(a, b, eps=1e-30)` and softmax-based `recall_probabilities(scores)` in `code/ms_tcm/similarity.py` per contracts/model-api.md and research R4

**Checkpoint**: Foundation ready — user story implementation can now proceed

---

## Phase 3: User Story 1 — Simulate cued recall for all four conditions (Priority: P1) 🎯 MVP

**Goal**: Run the MS-TCM forward simulator and reproduce the §4.4 analytical anchor (closed-form 0.86603 grouped / 0.80532 bridge; the PDF prints 0.866 / 0.806, the latter a rounding-cascade artifact) at 1e-6 tolerance on hand-derived inputs

**Independent Test**: `pytest code/tests/test_composite.py::test_section_4_4_numerical_anchor` asserts the closed-form values √0.75 = 0.86603 and 0.2·0.5625 + 0.8·√0.75 = 0.80532 at 1e-6 tolerance, given β_G = β_S = 0.5, w_G = 0.2, w_S = 0.8, m = 3

### Tests for User Story 1 (write first; they must fail before implementation lands)

- [X] T010 [P] [US1] Write `test_section_4_4_numerical_anchor` in `code/tests/test_composite.py` — hand-construct two unit-norm input vectors, step the context updates forward with β_G = β_S = 0.5, and assert composite similarity equals √0.75 = 0.86603 (grouped, one-step) and 0.2·0.5625 + 0.8·√0.75 = 0.80532 (bridge, m=3) at 1e-6 tolerance. Pins FR-013a, SC-002, notes/ms-tcm.pdf §4.4 (with the rounding-cascade caveat recorded in the test docstring)
- [X] T011 [P] [US1] Write `test_global_context_drift` and `test_storyline_context_frozen_when_inactive` in `code/tests/test_context.py` — assert `c_G(t+1) = ρ_G·c_G(t) + β_G·c_in` element-wise at 1e-12; assert `c_S(t+1) == c_S(t)` exactly when storyline S is inactive (pins FR-001, notes/ms-tcm.pdf §3.2)
- [X] T012 [P] [US1] Write `test_composite_weights_sum_to_one_enforced` and `test_retrieval_weights_independent` in `code/tests/test_composite.py` — assert `ValueError` raised when w_G + w_S ≠ 1 within 1e-12; assert retrieval weights can differ from encoding weights (pins FR-003, FR-004)
- [X] T013 [P] [US1] Write `test_cosine_similarity_zero_vector_returns_zero` and `test_recall_probabilities_sum_to_one` in `code/tests/test_similarity.py` — assert cosine_similarity returns 0.0 for zero-norm inputs (no NaN); assert softmax output sums to 1 ± 1e-12 (pins FR-005, research R4)
- [X] T014 [US1] Write `test_single_storyline_matches_standard_tcm` in `code/tests/test_model.py` — with K = 1, MS-TCM and standard TCM output per-event composite-context trajectories equal within 1e-12 (pins FR-006 + research R6 K=1 edge case)
- [X] T015 [US1] Write `test_list_boundary_resets_contexts` in `code/tests/test_model.py` — construct a two-list minimal dataset, run encoding, assert c_G and every c_S are zero at the first step of list 2 (pins spec Edge Cases)

### Implementation for User Story 1

- [X] T016 [US1] Implement `update_global_context(c_g_prev, beta_g, c_in) -> c_g` and `update_storyline_context(c_s_prev, beta_s, c_in) -> c_s` in `code/ms_tcm/context.py` per data-model.md §4.1 steps 1–3; compute ρ internally from β (FR-001, FR-002)
- [X] T017 [US1] Implement `compose_encoding(c_g, c_s, w_g, w_s) -> c_comp` and `compose_retrieval(c_g, c_s, w_g_ret, w_s_ret) -> c_ret` in `code/ms_tcm/composite.py` per data-model.md §3 step 4 and §4 step 1
- [X] T018 [US1] Implement `encode(dataset, parameters) -> EncodingState` in `code/ms_tcm/model.py` — iterate (participant, list, serial_position), reset c_G and all c_S to zero at each list boundary, apply T016/T017 updates; record `c_global`, `c_storyline`, `c_composite`, `active_storyline` arrays per data-model.md §3
- [X] T019 [US1] Implement `score_first_recall(state, participant, list_)` and `score_next_recall(state, participant, list_, last_recalled_serial_position)` in `code/ms_tcm/model.py` per data-model.md §4.2 — compute cosine similarities between the reinstated retrieval composite and every candidate's encoding composite, apply the optional λ-interference factor (exp(0)=1 when disabled), pass through `recall_probabilities`, return a length-W probability vector over the W presented positions. In free recall the candidate set is always the full list; NaN return is reserved for pathological inputs and the likelihood routine treats it as a hard error (pins FR-005)
- [X] T020 [US1] Implement `MSTCMModel` orchestrator methods in `code/ms_tcm/model.py` — public surface per contracts/model-api.md §4: `encode`, `score_first_recall`, `score_next_recall`. No separate `run(dataset)` convenience in v1; callers route through `dataset_log_likelihood` (for fitting) or `sample_recalls` (for synthetic data).
- [X] T021 [US1] Implement `ms_tcm.mechanisms` module — `apply_resumption_reinstatement(c_s, c_s_prev, gamma)` (§5.1), `apply_conversational_references(c_s, refs_in, alphas)` (§5.2), `interference_factor(lambda_val, i_ij)` (§5.3); every function is a no-op when its respective parameter is 0 / empty (FR-007)
- [X] T021b [US1] Implement `sample_recalls(model, dataset_skeleton, rng, *, recall_length_fn=None) -> pa.Table` in `code/ms_tcm/model.py` per contracts/model-api.md §10 — deterministically sample a `recalled.parquet`-shaped table from MS-TCM's per-step probabilities; used by the parameter-recovery test T042 and any future synthetic-data workflow (resolves U3; pins SC-009)

**Checkpoint**: User Story 1 is fully functional when the §4.4 anchor unit test passes and a two-list minimal in-memory dataset round-trips through encoding without NaNs

---

## Phase 4: User Story 2 — Load and inspect the canonical dataset format (Priority: P1) 🎯 MVP

**Goal**: Read the bundled FRFR-category dataset (or any future dataset of the same shape), validate against the seven FR-012 rules, and hand a typed `Dataset` to the simulator

**Independent Test**: `ms-tcm validate data/raw/frfr_category` returns exit code 0 with a summary matching `manifest.json`; running the same command on a mutated copy returns non-zero with the offending field named

### Tests for User Story 2 (write first)

- [X] T022 [P] [US2] Write `test_load_frfr_category_structure` in `code/tests/test_frfr.py` — load the bundled dataset, assert 7680 presented rows / 5215 recalled rows / 30 participants / 16 lists/pt / 16 words/list / exactly 4 categories per list (pins manifest.json row_counts and design)
- [X] T023 [P] [US2] Write seven malformed-dataset tests in `code/tests/test_schema.py` covering each FR-012 rule: `test_missing_required_file`, `test_wrong_column_type`, `test_orphan_recall_points_to_missing_presented`, `test_list_group_mismatches_list_index`, `test_duplicate_presented_row`, `test_manifest_hash_detects_tampering`, `test_manifest_malformed_json` (pins FR-012, SC-005)
- [X] T024 [P] [US2] Write `test_roundtrip_save_load_is_lossless` in `code/tests/test_dataset.py` — construct a small `Dataset` in memory, save it to a temp directory, load it back, assert all tables and metadata are identical cell-by-cell (pins FR-008)

### Implementation for User Story 2

- [X] T025 [US2] Implement `Dataset` frozen dataclass in `code/ms_tcm/dataset.py` — fields (`presented`, `recalled`, `manifest`); `__post_init__` schema check; convenience properties (`num_participants`, `num_lists_per_participant`, `num_words_per_list`, `categories_in_list(p, l)`, `feature_dim`)
- [X] T026 [US2] Implement `save_dataset(dataset, path)` in `code/ms_tcm/dataset.py` — write `presented.parquet` and `recalled.parquet` with deterministic Parquet settings (zstd level 1, no dictionary, sorted rows), write `presented.csv` and `recalled.csv`, write `manifest.json` via `io.write_json_sorted` with fresh SHA-256 hashes computed last
- [X] T027 [US2] Implement `load_dataset(path) -> Dataset` in `code/ms_tcm/dataset.py` — read Parquet via pyarrow, parse `manifest.json`, construct `Dataset`; verify manifest hashes against on-disk bytes and raise on mismatch
- [X] T028 [US2] Implement `load_frfr_category(path="data/raw/frfr_category") -> Dataset` in `code/ms_tcm/frfr.py` — thin convenience wrapper around `load_dataset` that also checks the dataset's shape matches Manning et al. 2023 expectations (30 pts, 16×16, 4 cats/list)
- [X] T029 [US2] Implement `validate_dataset(path) -> ValidationReport` in `code/ms_tcm/schema.py` — check all seven FR-012 rules, return a report naming the specific file/column/row for each violation; used by both `ms-tcm validate` and the library load path
- [X] T030 [US2] Implement the multi-hot feature encoder in `code/ms_tcm/features.py` — `encode_features(presented_df) -> np.ndarray`; each row becomes a concatenation of one-hots over category (15 + 1 unknown), size (small/large), first_letter (26 bins), length (binned 1–12), color (8 bins of RGB space), location (4 quadrants of the display). Dimensionality is recorded and used everywhere downstream; the encoder is pure (no hidden state) so the mapping is byte-deterministic across platforms

**Checkpoint**: User Story 2 closes the loop — `ms-tcm validate data/raw/frfr_category` exits 0, and the simulator from US1 can be handed a real `Dataset`

---

## Phase 5: User Story 3 — Reformat the raw FRFR data deterministically (Priority: P1) 🎯 MVP

**Goal**: Idempotent reformat script; bundled dataset survives a round trip; any reviewer on any platform can reproduce the committed files

**Independent Test**: `python scripts/reformat_frfr_category.py --egg <same egg>` twice back-to-back produces byte-identical Parquet, CSV, and `manifest.json`; SHA-256 hashes unchanged

### Tests for User Story 3 (write first)

- [X] T031 [P] [US3] Write `test_reformat_is_idempotent` in `code/tests/test_frfr.py` — if the upstream egg cache exists, invoke the reformat script twice with the same `--egg` argument into two temp directories, assert every output file's SHA-256 matches between runs (pins FR-011, SC-003). Skip gracefully if the cache is not present (so CI without network still passes; the local developer run is authoritative)
- [X] T032 [P] [US3] Write `test_shipped_dataset_hashes_match_manifest` in `code/tests/test_frfr.py` — compute SHA-256 of each committed file under `data/raw/frfr_category/` and compare against `manifest.json`; fail if any mismatch (this is the primary guard against accidental data drift in the repo)
- [X] T032b [P] [US3] Write `test_shipped_dataset_has_all_required_files` in `code/tests/test_frfr.py` — assert that `data/raw/frfr_category/` contains exactly the files enumerated in `manifest.json.files` plus `manifest.json` itself, and no extras (resolves C3; pins FR-010)
- [X] T033 [P] [US3] Write `test_manifest_hashes_match_file_contents` in `code/tests/test_schema.py` — generic round-trip test on any dataset that passes `validate_dataset`: load, rewrite with `save_dataset`, reload, assert the new manifest's hashes match the rewritten files (exercises the whole save/load/hash contract)

### Implementation for User Story 3

- [X] T034 [US3] (Already done) The script `scripts/reformat_frfr_category.py` is committed. Audit it against the final contracts: verify it (a) respects the schemas defined in T025/T026, (b) writes identical manifest keys, (c) includes a module-level docstring that justifies its use of the legacy in-band string decoder (plan.md Complexity Tracking row 2). If any drift: update the script to match the contracts, re-run, commit the new `data/raw/frfr_category/` output
- [X] T035 [US3] Add a smoke-run target to `Makefile` (or equivalent cross-platform invocation documented in quickstart.md) for `python scripts/reformat_frfr_category.py`; this is the documented way a reviewer regenerates the bundled dataset. Not a library import; only the CLI command changes
- [X] T036 [US3] Document the FRFR provenance, idempotence guarantee, and SHA-256 verification procedure in `data/README.md` — replace the template text with content specific to the `data/raw/frfr_category/` layout and the upstream egg citation

**Checkpoint**: Running `pytest code/tests/test_frfr.py::test_shipped_dataset_hashes_match_manifest` green and `pytest code/tests/test_schema.py::test_manifest_hashes_match_file_contents` green on a fresh clone with no network — the committed dataset is self-verifying

---

## Phase 6: User Story 4 — Fit MS-TCM with MLE + 95 % bootstrap CIs (Priority: P1)

**Goal**: A single CLI command fits MS-TCM (or standard TCM) to the FRFR-category dataset and reports per-parameter MLE + 95 % CI, plus log-likelihood / AIC / BIC for model comparison

**Independent Test**: `ms-tcm fit data/raw/frfr_category --out <dir>` completes in under 10 min with 1000 bootstrap reps and emits `fit_summary.json` with finite `{mle, ci_lower, ci_upper}` for every parameter

### Tests for User Story 4 (write first)

- [X] T037 [P] [US4] Write `test_likelihood_matches_hand_derivation_3word` in `code/tests/test_likelihood.py` — construct a minimal 3-word, 2-category hand-derived list; compute expected per-transition log-probability manually (with numbers written out in a comment block); assert the library's `list_log_likelihood` matches within 1e-12 (pins FR-017, Constitution I)
- [X] T038 [P] [US4] Write `test_optimizer_recovers_simple_params` in `code/tests/test_fit.py` — on a very small hand-generated dataset (K=2, 2 lists, 4 words/list) where the analytical optimum is known, assert the optimizer finds the MLE within 1e-6 (pins research R7; a sanity check that L-BFGS-B is wired correctly)
- [X] T039 [P] [US4] Write `test_bootstrap_is_deterministic_same_seed` in `code/tests/test_bootstrap.py` — run the bootstrap twice on the same dataset with the same seed and a tiny n_bootstraps (e.g. 10) using a 3-participant subset; assert both runs produce the same bootstrap indices and the same per-draw MLE arrays to 1e-12 (pins FR-018, SC-003 generalized)
- [X] T040 [P] [US4] Write `test_ci_contains_mle_and_is_nontrivial` in `code/tests/test_bootstrap.py` — on any fit, assert `ci_lower ≤ mle ≤ ci_upper` and `ci_upper > ci_lower` for every parameter (pins FR-018)
- [X] T041 [US4] Write `test_standard_tcm_flag_constrains_wstoryline_to_zero` in `code/tests/test_fit.py` — fit with `--standard-tcm` on a small dataset; assert the output's `w_storyline.mle == 0.0` exactly and that the bootstrap CI on `w_storyline` is `[0.0, 0.0]` (pins FR-020)
- [X] T041b [US4] Write `test_fit_summary_contains_required_keys` in `code/tests/test_fit.py` — call `FitResult.save()` on a small fit, reload the JSON, assert every required key from contracts/model-api.md §9 is present with the right type (resolves C6; pins FR-019)
- [ ] T042 [US4] Write `test_parameter_recovery_on_synthetic_recalls` in `code/tests/test_parameter_recovery.py` — call `sample_recalls` (T021b) to generate synthetic recall sequences from MS-TCM at a known parameter vector, fit, assert ≥ 95 % of parameters have their true value inside their bootstrap 95 % CI (pins SC-009). Runs with a small number of bootstraps (e.g. 200) and a single synthetic "participant pool" for CI speed; the full 1000-rep run is a manual validation, not a CI test

### Implementation for User Story 4

- [X] T043 [US4] Implement `list_log_likelihood(presented_list, recalled_list, parameters, encoding_state) -> float` in `code/ms_tcm/likelihood.py` — iterate observed recall transitions, evaluate the softmax probability of each observed target given the cue and the current context state, sum log-probabilities. Extra-list intrusions (serial_position == 0) do not contribute to the likelihood but their count is recorded for diagnostics
- [X] T044 [US4] Implement `dataset_log_likelihood(dataset, parameters) -> float` in `code/ms_tcm/likelihood.py` — sum `list_log_likelihood` across every (participant, list); this is the objective the optimizer minimizes (negated)
- [X] T045 [US4] Implement `fit_mle(dataset, *, n_restarts=5, seed, standard_tcm=False, bounds=None) -> FitResult` in `code/ms_tcm/fit.py` — reparameterize the constrained weights (w_G, w_S with sum=1) via a single unconstrained logit and the β values via logistic transforms, run scipy's L-BFGS-B `n_restarts` times from seeded random initial points, take the best log-likelihood; return a `FitResult` with the MLE parameter vector and convergence diagnostics
- [X] T046 [US4] Implement `bootstrap_ci(dataset, *, n_bootstraps=1000, seed, standard_tcm=False, ci=0.95) -> dict[str, dict]` in `code/ms_tcm/bootstrap.py` — resample participants with replacement (seeded), refit `fit_mle` on each resample, collect per-parameter estimates, return percentile CIs at `ci`. Discard and flag non-convergent draws; abort with an informative error if fewer than 90 % converge
- [X] T047 [US4] Implement `FitResult.save(out_dir)` — write `fit_summary.json` (sorted JSON) with per-parameter `{mle, ci_lower, ci_upper, ci_method, n_bootstraps, n_converged}` plus fit-level metadata (log-likelihood, AIC, BIC, n_participants, n_lists, n_recalls_used, seed, elapsed_seconds, ms_tcm_version, dataset_manifest_sha256), and `fit_bootstrap.parquet` with every bootstrap draw's parameter estimates
- [X] T048 [US4] Wire `ms-tcm fit <dataset-dir> --out <dir>` in `code/ms_tcm/cli.py` — flags for `--seed`, `--n-bootstraps`, `--n-restarts`, `--standard-tcm`, `--gamma/--alpha/--lambda` to enable optional §5 mechanisms; exit-code conventions per contracts/cli.md
- [X] T049 [US4] Demo notebook at `code/notebooks/demo.ipynb` produces one figure (e.g. MS-TCM vs standard-TCM lag-CRP on FRFR-category) using only `from ms_tcm import …` imports — no duplicated definitions (Constitution II)

**Checkpoint**: `ms-tcm fit data/raw/frfr_category --out data/processed/fits/mstcm` produces a populated `fit_summary.json` with finite MLE and 95 % CI for every parameter, and the same command with `--standard-tcm` produces an equivalent summary for the baseline

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that touch multiple user stories; run after all four are green

- [X] T050 [P] Implement the `ms-tcm` CLI dispatcher in `code/ms_tcm/cli.py` — argparse-based subcommands `validate`, `run`, `fit` per contracts/cli.md; exit codes per contracts/cli.md §5
- [X] T051 [P] Update `code/README.md` to describe the notebook ↔ figure mapping (1:1 per CLAUDE.md), the CLI quickstart (validate → fit), and the `ms_tcm` import surface
- [X] T052 [P] Update the top-level `README.md` to replace the CDL-template "Paper title" placeholder with a project-specific paragraph pointing at `notes/ms-tcm.pdf`, `data/raw/frfr_category/`, and the two tracked GitHub issues (#1 synthetic generator, #2 paper)
- [ ] T053 Run `ms-tcm validate data/raw/frfr_category` + `ms-tcm fit data/raw/frfr_category --out data/processed/fits/mstcm --n-bootstraps 200 --seed 42` end-to-end on macOS, paste the resulting `fit_summary.json` MLEs into `code/tests/test_cross_platform.py` as a regression fixture; repeat the same commands on Ubuntu and Windows in CI and fail if the MLEs disagree by more than 1e-10
- [X] T053b Add `.github/workflows/ci.yml` with a matrix of `macos-latest`, `ubuntu-latest`, `windows-latest` running `pip install -e ".[dev]"` + `pytest code/tests` + `ms-tcm validate data/raw/frfr_category` + `ms-tcm fit data/raw/frfr_category --out runs/ci --n-bootstraps 50 --seed 42 --force`; fail if the cross-platform regression fixture from T053 disagrees by more than 1e-10. Resolves C5; pins FR-015.
- [X] T054 Verify the Constitution Check once more before handoff: (a) `pytest code/tests` green; (b) zero duplicated function definitions across `code/ms_tcm/` **and** `code/notebooks/` — use a script at `scripts/check_no_duplicate_defs.py` that extracts top-level `def` names from `.py` files and top-level `def` names from notebook code cells, checks that every notebook-defined name is either absent from the package or a trivial `from ms_tcm import …` re-export, and fails with a named offender otherwise; (c) every numerical assertion cites a section of `notes/ms-tcm.pdf`, a row in `manifest.json`, or a Manning et al. (2023) reference; (d) `pip install -e .` + `ms-tcm validate` + `ms-tcm fit` pass on all three platforms. Resolves C4.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup; **BLOCKS** all user stories
- **US1 (Phase 3)**, **US2 (Phase 4)**, **US3 (Phase 5)** are all P1 and independent once Foundational lands. US2 needs US1 for end-to-end integration tests, and US3 verifies the data US2 consumes — they're usually worked in the order US1 → US2 → US3 but parallel work is possible between US1 and US3
- **US4 (Phase 6)**: Depends on US1 (simulator) + US2 (dataset load) + US3 (bundled data known-good)
- **Polish (Phase 7)**: Depends on all four user stories

### User Story Dependencies

- **US1 (P1)**: After Foundational
- **US2 (P1)**: After Foundational; integrates with US1 at T020 (simulator run on a real Dataset)
- **US3 (P1)**: After Foundational; independent of US1/US2 since the reformat script and the shipped data are already present — US3 is primarily about pinning the guarantees
- **US4 (P1)**: After US1, US2, US3

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- `params.py` / `io.py` (Foundational) before anything that consumes them
- `context.py` → `composite.py` → `model.py` (US1 order)
- `schema.py` → `dataset.py` → `frfr.py` (US2 order)
- `likelihood.py` → `fit.py` → `bootstrap.py` (US4 order)
- Story complete before moving on

### Parallel Opportunities

- All [P]-marked Setup tasks (T003, T004, T005) can run in parallel
- All [P]-marked Foundational tasks (T008, T009) can run in parallel
- All US1 test tasks (T010–T013) can run in parallel; T014/T015 are sequential as they depend on the foundational params/context code
- All US2 test tasks (T022, T023, T024) can run in parallel
- All US3 test tasks (T031, T032, T033) can run in parallel
- All US4 test tasks (T037–T041) can run in parallel; T042 (parameter recovery) depends on the simulator + likelihood machinery
- Polish tasks T050, T051, T052 can run in parallel

---

## Parallel Example: User Story 1 tests

```bash
# Launch US1 unit tests in parallel:
Task: "Write test_section_4_4_numerical_anchor in code/tests/test_composite.py"
Task: "Write test_global_context_drift + test_storyline_context_frozen_when_inactive in code/tests/test_context.py"
Task: "Write test_composite_weights_sum_to_one_enforced + test_retrieval_weights_independent in code/tests/test_composite.py"
Task: "Write test_cosine_similarity_zero_vector_returns_zero + test_recall_probabilities_sum_to_one in code/tests/test_similarity.py"
```

---

## Implementation Strategy

### MVP First (US1 + US2 + US3 together)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3 (US1) + Phase 4 (US2) + Phase 5 (US3)
4. **STOP and VALIDATE**: `pytest code/tests` green; `ms-tcm validate data/raw/frfr_category` exits 0; the simulator runs on the bundled dataset without NaNs
5. This is the MVP: the §4.4 anchor is pinned, the shipped dataset is self-verifying, and the simulator consumes it end-to-end

### Incremental Delivery

1. Setup + Foundational + US1 + US2 + US3 → MVP
2. Add US4 → MLE + 95 % CI fitter and standard-TCM baseline comparison; this is the scientific payoff of the feature
3. Polish → Cross-platform CI parity, docs, constitution re-check
4. Close out issue #1 (synthetic generator) and issue #2 (paper) — these are tracked but out of scope for this feature

### Parallel Team Strategy

- US1 (model), US2 (dataset format), US3 (reformat guarantees) can be staffed by three people concurrently after Foundational completes; they meet at T020 / T029 / T034
- US4 must follow because it depends on all three; once it starts, the fitter (T043–T047) and CI wiring (T048) can be split across two developers

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps a task to a specific user story for traceability
- Tests MUST be written and verified to fail before implementation code lands (Constitution Principle I)
- Commit after each task or logical group; outstanding changes committed before any in-place rewrite (Constitution II)
- Every numerical assertion MUST cite the section of `notes/ms-tcm.pdf`, the Manning et al. (2023) reference, or the row of `manifest.json` it anchors (Constitution III + SC-008)
- Stop at any checkpoint to validate the story independently
- Avoid: duplicated function definitions, mocks, placeholder values, fallbacks that silently substitute for real computation
- The synthetic-dataset work deferred to [issue #1](https://github.com/ContextLab/ms-tcm/issues/1) and the paper writeup tracked in [issue #2](https://github.com/ContextLab/ms-tcm/issues/2) are NOT part of this feature. They reference this feature's design artifacts so they can pick up cleanly when they are scheduled
