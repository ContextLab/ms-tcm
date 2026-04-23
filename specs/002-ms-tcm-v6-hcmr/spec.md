# Feature Specification: MS-TCM v6 (Hierarchical CMR) Rewrite + Fast Inference + Paper/Docs Sync

**Feature Branch**: `002-ms-tcm-v6-hcmr`
**Created**: 2026-04-22
**Status**: Draft
**Input**: User description (abridged): Overhaul MS-TCM to the v6 hierarchical-CMR design (`notes/two_level_cmr_v6.pdf`), inheriting from Cornell & Zhang 2025 (`notes/CornZhan25.pdf`) and adding one new parameter (λ, storyline-return reinstatement at encoding). The current v1 implementation fails to reproduce standard free-recall phenomena (serial position curve, probability of first recall, lag-CRP) and fits slowly (~10 min per 1000-bootstrap run on FRFR-category). This feature replaces the v1 math, adds behavioral regression tests that must pass before release, optimizes inference in three measured tiers, and propagates the changes through the paper, bibliography, and repository documentation.

## Overview

This feature delivers a ground-up replacement of the MS-TCM model implementation to match the v6 design notes, aligns the model with the published Cornell & Zhang 2025 hierarchical CMR framework, adds first-class regression tests for the free-recall benchmark phenomena (SPC, p(first recall), lag-CRP), delivers a measurable ≥ 5× (ideally ≥ 20×) fit-time speedup, and synchronizes the LaTeX paper and all project-level documentation with the new formulation. A researcher cloning the repository after this feature ships should find a model that (a) reproduces the canonical free-recall behavioral signatures out of the box on FRFR-category, (b) fits in minutes rather than hours, (c) is described in the paper, the `CLAUDE.md`, the READMEs, and the `specs/` contracts using the v6 parameter inventory (β_enc, β_story, γ_fc, k, λ, plus β_rec and ε_d for free-recall paradigms), and (d) correctly cites Cornell & Zhang 2025 as the direct theoretical predecessor.

## Clarifications

### Session 2026-04-23

- Q1 (scope): v1 model code retirement strategy → **Destructive delete.** The obsolete v1-only modules (`composite.py`, the v1 parts of `context.py`, `mechanisms.py`, and v1-specific fields in `params.py`), the v1-only tests (including `test_section_4_4_numerical_anchor` and any that reference `w_global + w_storyline = 1`), and any v1-only notebooks are removed outright. Historical interpretation is preserved in `notes/v6_migration.md` (FR-047) plus git history; git blame continues to show the deleted content.
- Q2 (test tolerance): Behavioral-regression tolerance → **Two-layer test combining (B) qualitative shape and (C) best-fit against FRFR-category human data.** Layer 1 is a set of shape assertions that must pass before the model is considered correct at all: SPC has a detectable primacy limb (P(recall) at position 1 > P(recall) at middle position) AND a detectable recency limb (P(recall) at final position > P(recall) at middle position); pFR is maximized in the second half of the list (argmax > W/2); lag-CRP peaks at lag = +1 (argmax(CRP) among non-zero lags equals +1) with forward asymmetry (CRP[+1] > CRP[-1]) and CRP[+1] > CRP[+2]. Layer 2 is a relative-fit comparison: the MS-TCM-simulated SPC, pFR, and lag-CRP curves match the FRFR-category human curves with a per-bin relative error ≤ 20 %, AND MS-TCM fits no worse than the `--standard-tcm` reduction on the same three curves under the same relative-error metric. The test file is `code/tests/test_behavioral_regression.py`; reference human curves are computed from `data/raw/frfr_category/recalled.parquet` and cached in `data/processed/reference_curves/frfr_category_{spc,pfr,lag_crp}.parquet` with SHA-256s recorded in a dedicated `data/processed/reference_curves/manifest.json` (NOT the FRFR-category raw dataset's `manifest.json`, which stays byte-identical per FR-061). [A1 resolved]
- Q3 (cross-platform tolerance for Tier 2): JAX numerical precision → **Offer both, float64 is the default.** An opt-in knob (environment variable `MS_TCM_JAX_DTYPE=float32` or the equivalent `--jax-dtype` flag) selects float32 when maximum speed is wanted; float64 is the default and is held to the Tier 1 1e-10 cross-platform tolerance bar. Opt-in float32 path is held to 1e-8. Both dtypes run in CI as a test-matrix dimension; published runs in the paper use default (float64).

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Model reproduces free-recall benchmark phenomena on FRFR-category (Priority: P1)

A memory researcher runs the new fitter on the bundled FRFR-category dataset and, using the fitted parameters, regenerates three canonical free-recall plots from the simulated recall output: a serial position curve (SPC), a probability of first recall (pFR), and a lag-conditional response probability (lag-CRP). Without any hand-tuning, the model produces the qualitative signatures that Cornell & Zhang 2025 (Figure 2) and Kahana et al. (2002) report for their free-recall datasets: a U-shaped SPC with primacy and recency, a pFR weighted toward the end of the list, and a lag-CRP that peaks at +1 with a forward asymmetry.

**Why this priority**: without this, the rewrite has not solved the core problem that triggered it. The paper's claim that MS-TCM extends standard CMR is only credible if the model matches CMR's behavior on the benchmark phenomena that CMR is known to reproduce.

**Independent Test**: a dedicated behavioral regression test module runs `ms-tcm fit data/raw/frfr_category` at a small-but-real configuration (≥ 3 restarts, ≥ 50 bootstraps), uses the fitted MLE to simulate recalls via the public `sample_recalls` API, computes SPC / pFR / lag-CRP from the simulated recalls, and asserts the outputs pass the tolerance policy resolved in Q2 against reference curves. The test is named, cited to `notes/CornZhan25.pdf` Figure 2 and/or the FRFR-category manifest, and runs in under 3 minutes on CI.

**Acceptance Scenarios**:

1. **Given** the bundled FRFR-category dataset and the fitted MS-TCM parameters from a fresh fit, **When** `sample_recalls` simulates recalls and SPC/pFR/lag-CRP are computed, **Then** all three curves pass the Q2-resolved tolerance policy.
2. **Given** a standard-TCM (MS-TCM reduction with λ=0, storyline context disabled) fit on the same data, **When** the same three behavioral curves are computed, **Then** they also pass the Q2-resolved tolerance policy (standard CMR is known to reproduce these; a failure here indicates a CMR-layer bug, not an MS-TCM extension bug).
3. **Given** a regression suite run, **When** any of the three curves fails its tolerance, **Then** the test surfaces which curve failed, on what condition, and against which reference, with a one-line pointer to the relevant figure in `notes/CornZhan25.pdf` or an entry in the FRFR-category manifest.

---

### User Story 2 — Fit wall-clock drops to under 2 minutes (Tier 1) and, optionally, 30 seconds (Tier 2) (Priority: P1)

A researcher iterating on model design runs `ms-tcm fit data/raw/frfr_category --out <dir> --n-bootstraps 1000 --n-restarts 5 --seed 42` and the fit completes in under 2 minutes on a modern laptop (Tier 1). If Tier 2 (JAX) is subsequently enabled, the same command completes in under 30 seconds. The same CLI works on macOS, Ubuntu, and Windows without conditional switches exposed to the user.

**Why this priority**: the whole point of the performance tier work is to unblock rapid design iteration. A rewrite that is correct but still takes 10 minutes per fit will not be used and will not produce the iterations needed for the paper.

**Independent Test**: a performance benchmark script (`scripts/benchmark_fit.py`) runs the standard fit command with default settings, records wall-clock time, and a CI job asserts elapsed seconds is below the tier gate. The benchmark is deterministic (fixed seed), and regressions are reported as build failures on PRs that touch fit-relevant code.

**Acceptance Scenarios**:

1. **Given** the bundled FRFR-category dataset, **When** `ms-tcm fit --n-bootstraps 1000 --n-restarts 5 --seed 42` is run after the Tier 1 optimizations land, **Then** the fit completes in under 120 seconds wall-clock on a modern laptop (baseline: current ~600 s; ≥ 5× speedup).
2. **Given** Tier 1 optimizations applied, **When** the same command is run on a 20-participant synthetic subset, **Then** all results from the new code agree with pre-optimization reference MLE + CI values to within the Q3-resolved cross-platform tolerance (no silent correctness regression).
3. **Given** Tier 2 (JAX) is opted in via a config or environment flag, **When** the same command is run, **Then** the fit completes in under 30 seconds (≥ 20× speedup) and all behavioral regression tests (User Story 1) still pass.
4. **Given** a Tier 2 JAX fit, **When** the same seed is used twice, **Then** MLEs agree to within 1e-10 on the default float64 dtype, or 1e-8 on the opt-in float32 dtype (Q3).

---

### User Story 3 — Paper and documentation accurately describe the v6 model (Priority: P1)

A reader opens `paper/main.tex`, reads §3 (Model) and §4 (Derivation), and finds the hierarchical CMR formulation from `notes/two_level_cmr_v6.pdf`: two context levels (item, storyline), boundary synchronization, storyline caching in M^SC, storyline-return reinstatement (λ, Eq 6), and the parameter inventory (β_enc, β_story, γ_fc, k, λ, plus β_rec, ε_d for free-recall). The reader sees Cornell & Zhang 2025 cited as the direct theoretical predecessor, with an explicit statement that MS-TCM adds exactly one mechanism (λ) on top of their hierarchical CMR. The v1 §4.4 numerical anchor is absent from the paper body. A brief note explains the pre-experimental context matrix $M^{FC}_{\text{pre}}$ (identity for word lists, optional USE embeddings for naturalistic stimuli). The reader also opens `CLAUDE.md`, `README.md`, `code/README.md`, and `data/README.md` and finds references to `notes/two_level_cmr_v6.pdf` as the canonical spec, correct parameter names, and an accurate description of the fit workflow. A "v6 vs. v1" section (either in `paper/supplement.tex` or a `notes/v6_migration.md` file) explains why the rewrite was necessary.

**Why this priority**: the paper is the primary scientific artifact. A correct implementation in the repo, combined with an outdated paper, would confuse reviewers and make the `cd paper && ./compile.sh` workflow (which produces the committed `main.pdf` / `supplement.pdf`) misleading.

**Independent Test**: (a) a script at `scripts/check_paper_consistency.py` greps the built `main.tex` for the retired v1 symbols (β_G, β_S, w_G, w_S, composite similarity "0.866", "0.806", "0.80532") and fails on any match; (b) the same script checks that `CornZhan25` bibkey appears in `paper/CDL-bibliography/` or a local bib file and is cited in `main.tex`; (c) the script verifies `paper/main.tex` compiles cleanly via `cd paper && ./compile.sh` without warnings about undefined references; (d) a documentation checklist manually verifies CLAUDE.md, the three READMEs, and the contracts/ quickstart/ data-model files mention v6 and not v1 as the canonical reference.

**Acceptance Scenarios**:

1. **Given** the updated repository, **When** `paper/compile.sh` runs, **Then** `main.pdf` and `supplement.pdf` are produced without bibliography errors or undefined-reference warnings, and the body text contains the v6 parameter names and Cornell & Zhang 2025 citation.
2. **Given** the updated repository, **When** `scripts/check_paper_consistency.py` runs, **Then** it exits 0 (zero v1-specific artifacts in the paper body).
3. **Given** the updated repository, **When** a reader opens `CLAUDE.md`, **Then** the "Project" section points at `notes/two_level_cmr_v6.pdf` as the canonical spec (not `notes/ms-tcm.pdf`), and the parameter regimes list uses v6 symbols.
4. **Given** the updated repository, **When** a reader opens `README.md`, `code/README.md`, and `data/README.md`, **Then** the descriptions match the new CLI surface, parameter names, and fit workflow.

---

### User Story 4 — Pluggable pre-experimental context matrix for cued-recall readiness (Priority: P2)

A researcher who plans to later fit the Xu et al. 2026 naturalistic cued-recall data loads an `MFCPreMatrix` object from USE sentence embeddings of scene annotations and hands it to the model without touching any other code. The same `ms-tcm fit` CLI accepts a `--pre-context identity|use-embeddings` flag that selects between the default identity matrix (used for the FRFR-category worked example) and a loader for pre-computed embeddings.

**Why this priority**: the v6 notes §1.5 explicitly calls out the identity-vs-embeddings diagnostic as essential for claiming the λ mechanism is theoretically necessary. Shipping the rewrite without the hook would force a second feature before any naturalistic fit could happen.

**Independent Test**: a unit test at `code/tests/test_pre_context.py` constructs a small dummy embeddings matrix, loads it via the `MFCPreMatrix` API, runs the encoder on a minimal synthetic list, and asserts the resulting item-context trajectories differ from the identity-matrix case. No actual USE dependency is introduced in this feature (USE integration lands when the Xu et al. 2026 dataset does); the hook is just an interface.

**Acceptance Scenarios**:

1. **Given** a small dummy embeddings matrix supplied via the public API, **When** the model encodes a list, **Then** the item-context trajectories are a mixture of pre-experimental and experimental contexts per v6 Eq 1.5.1 with weight γ_fc.
2. **Given** no pre-experimental matrix is supplied, **When** the model encodes a list, **Then** the pre-experimental matrix defaults to identity (the FRFR-category case), behavior matches the current free-recall fit, and the γ_fc parameter remains identifiable.

---

### User Story 5 — Parameter starting values cite Cornell & Zhang 2025 Table 1 (Priority: P2)

A researcher reading `code/ms_tcm/params.py` and the paper's Methods section finds that the default starting values for the fitter are β_list=0.400, β_rein=0.300, β_enc=0.679, β_rec=0.326, γ_fc=0.315, k=6.50, ε_d=1.04, β_post=0.85, with a citation to Cornell & Zhang 2025 Table 1. The λ parameter is initialized at 0.8 per v6 §5.

**Why this priority**: inheriting documented starting values from a published model reduces the probability that the optimizer gets stuck in a local minimum that's far from any psychologically-interpretable regime. It also makes the paper's claim ("we inherit CMR values and add λ") verifiable at the code level.

**Independent Test**: a unit test at `code/tests/test_param_defaults.py` constructs a default `ModelParameters` instance and asserts each parameter equals the value from C&Z Table 1 (or v6 §5 for λ) to exact equality.

**Acceptance Scenarios**:

1. **Given** a default `ModelParameters()` call with no arguments, **When** its fields are inspected, **Then** they match the values above to exact equality, and each field has a docstring comment citing the C&Z Table 1 page number.

---

### Edge Cases

- **Standard-TCM reduction**: when λ=0 and the storyline context is disabled (one storyline only, or `--standard-tcm` flag), MS-TCM MUST reduce exactly to standard CMR with β_story drift (β_enc for the item layer, β_story for the single remaining context layer), and the behavioral regression test MUST still pass. A failure here indicates a bug in the reduction, not in MS-TCM. [A9 resolved: replaced v1 symbol β_list with v6 β_story]
- **Missing storyline labels**: for the FRFR-category dataset, category maps to storyline and all words have a category. For future cued-recall datasets that lack explicit storyline labels, the model MUST raise an explicit error rather than silently treating the entire list as one storyline.
- **Empty recall list**: a participant who fails to recall any word on a list contributes zero to the likelihood (not NaN), same as v1.
- **Intrusions**: extra-list intrusions (serial_position == 0) MUST continue to be counted separately and excluded from the lag-CRP likelihood, same as v1.
- **List boundary reset**: both the item-level and storyline-level contexts MUST reset to the reserved list-start vector `e_start` at each list boundary, preserving the TCM unit-norm invariant.
- **Tier-2 JAX fallback**: if JAX is not installed, the fitter MUST fall back to Tier 1 with a clear log message rather than failing.
- **Optimizer non-convergence**: non-convergent bootstrap draws are discarded and flagged; overall fit aborts if fewer than 90 % converge — unchanged from v1.
- **Retired v1 test**: `test_section_4_4_numerical_anchor` is deleted; the v1 composite-weight `w_global + w_storyline = 1` constraint is removed from `ModelParameters`. Any tests that referenced these must be rewritten or deleted.

## Requirements *(mandatory)*

### Functional Requirements — Model (v6 hierarchical CMR)

- **FR-001**: The model MUST implement the two-level drift from v6 §2.1: an item-level context drifting at β_enc and a storyline-level context drifting at β_story, with β_enc > β_story enforced at parameter validation.
- **FR-002**: At event boundaries within a storyline (v6 §2.2), the item-level context MUST synchronize to the storyline-level context: c_item ← c_story. Event boundaries do NOT have their own associative matrix.
- **FR-003**: At storyline switches (v6 §2.3), the outgoing storyline context MUST be cached to M^SC via ΔM^SC = g_{s(i-1)} · c^story_out^T, and the item-level context MUST synchronize to the new storyline context.
- **FR-004**: At storyline returns (v6 §2.4), the storyline context MUST be reinstated via c^story ← λ · c̃^story_{s(i)} + (1-λ) · c^story_{i-1}, where c̃^story_{s(i)} is read from M^SC. This is the sole new mechanism introduced by MS-TCM over Cornell & Zhang 2025.
- **FR-005**: The retrieval route MUST be the standard CMR item-level readout: a_j = (M^IC)^T · c^item_cue, then P(j | cue) = softmax(k · a_j). No task-instruction-dependent reweighting at retrieval (v6 §3.1–3.2: retrieval is instruction-blind; instructions are scoring filters).
- **FR-006**: Free-recall paradigms MUST activate β_rec (within-trial retrieval drift) and ε_d (stopping rule) per Cornell & Zhang 2025 Eqs 3 and 7. Cued-recall paradigms (Xu et al. 2026) do NOT use these.
- **FR-007**: The pre-experimental matrix M^FC_pre MUST be a first-class configurable object: default identity matrix (v6 §1.5 Option 1), with an API hook for loading pre-computed embedding matrices (v6 §1.5 Option 3). The γ_fc parameter MUST weight the mixture c^IN_i = (1 - γ_fc) · M^FC_pre · f_i + γ_fc · M^FC_exp · f_i per v6 Eq 1.5.1.
- **FR-008**: The model MUST enforce ρ = √(1 - β²) internally for each context level so that ||c||=1 is preserved at every step under unit-norm inputs.
- **FR-009**: Default parameter starting values in `ModelParameters` MUST match Cornell & Zhang 2025 Table 1 (β_list=0.400, β_rein=0.300, β_enc=0.679, β_rec=0.326, γ_fc=0.315, k=6.50, ε_d=1.04, β_post=0.85) and v6 §5 (λ=0.80). Each field's docstring MUST cite the source page or section. [A11 resolved: pinned as `0.80` two-decimal form matching C&Z Table 1 style; test and docstring MUST use the same string form]
- **FR-010**: At list boundaries, both c^item and c^story MUST reset to the reserved list-start unit vector `e_start` (one-hot at feature dimension 0).
- **FR-011**: The model MUST provide a `--standard-tcm` reduction that sets `λ=0` AND an explicit `standard_tcm: bool = False` field on `ModelParameters` that, when `True`, causes `HierarchicalCMRModel` to skip all storyline-level updates (drift, boundary caching, reinstatement) and treat the entire list as a single storyline. The reduction reduces MS-TCM to standard CMR — the single-level baseline required for paper-level model comparison. `ModelParameters.standard_tcm()` constructs this configuration; `--standard-tcm` on the CLI is the user-facing switch. [A18 resolved: the "marker" in data-model.md §1 is now this explicit `standard_tcm` bool field, not just λ=0]
- **FR-012**: The model MUST expose (as public API) a `sample_recalls(model, dataset_skeleton, rng, *, recall_length_fn=None)` function that deterministically samples a `recalled.parquet`-shaped table from fitted MS-TCM probabilities, for use by the behavioral regression tests and any future simulation.
- **FR-013**: The v1 model code is **destructively deleted** (Q1). Specifically: remove `code/ms_tcm/composite.py`, remove v1-specific helpers in `code/ms_tcm/context.py` and `code/ms_tcm/mechanisms.py`, remove v1-specific fields in `code/ms_tcm/params.py` (`w_global`, `w_storyline`, `w_global_ret`, `w_storyline_ret`, `gamma` in its v1-resumption meaning, `lambda_interference` in its v1-§5.3 meaning), remove the v1 tests (`test_section_4_4_numerical_anchor` and any asserting `w_global + w_storyline == 1`), and remove any notebook cells referencing the v1 symbols β_G, β_S, w_G, w_S, w_G^ret, w_S^ret. A migration note in `notes/v6_migration.md` (FR-047) MUST record the mapping from retired v1 symbols to v6 symbols; git history preserves the deleted content for any reader who wants to recover it.

### Functional Requirements — Behavioral regression

- **FR-020**: A test module at `code/tests/test_behavioral_regression.py` MUST compute SPC, pFR, and lag-CRP from recalls simulated by the fitted MS-TCM on FRFR-category and assert each curve against the **two-layer tolerance policy (Q2)**:
    - **Layer 1 (qualitative shape)**: SPC has a detectable primacy limb (`P(recall)[pos=1] > P(recall)[pos=W/2]`) AND a detectable recency limb (`P(recall)[pos=W] > P(recall)[pos=W/2]`); pFR has `argmax > W/2`; lag-CRP peaks at lag = +1 (`argmax over non-zero lags == +1`) with `CRP[+1] > CRP[-1]` and `CRP[+1] > CRP[+2]`.
    - **Layer 2 (relative fit)**: the MS-TCM-simulated curves match the FRFR-category empirical curves with per-bin relative error ≤ 20 %, AND MS-TCM's per-curve relative error is ≤ the `--standard-tcm` reduction's error on the same curve. Reference empirical curves are cached in `data/processed/reference_curves/frfr_category_{spc,pfr,lag_crp}.parquet`, with SHA-256s recorded in the FRFR-category manifest.
  Every numerical comparison in the test MUST cite Cornell & Zhang 2025 Figure 2, Kahana et al. 2002 Figure 1, the v6 notes section, or the FRFR-category manifest.
- **FR-021**: The same behavioral regression tests MUST pass under the `--standard-tcm` reduction (the CMR baseline is known to reproduce these; a failure there indicates a CMR-layer bug). Under the reduction, Layer 2 comparison is against self (MS-TCM vs. standard-TCM with the same fit) — it is the *MS-TCM* run's job to fit no worse than the standard-TCM baseline, not the other way around.
- **FR-022**: The existing parameter-recovery test (T042 in the 001 feature) MUST still pass in its "@slow" form on the new model (mapped onto the v6 parameter set: β_enc, β_story, γ_fc, k, λ, β_rec, ε_d).
- **FR-023**: Cross-platform numerical tolerance for derived outputs (MLEs, log-likelihoods) MUST be **1e-10 absolute tolerance** on the Tier 1 path. On the Tier 2 (JAX) path: **1e-10 in the default float64 dtype**, **1e-8 in the opt-in float32 dtype**. Both dtypes run in CI as a matrix dimension; paper-reported fits use the default float64.

### Functional Requirements — Performance

- **FR-030**: Tier 1 (Python-only): `encode_features` MUST be vectorized (no pandas `iterrows`), the pre-experimental features and M^FC_pre MUST be cached per dataset across likelihood evaluations, and the bootstrap loop MUST be parallelized via `multiprocessing.Pool`. Target: ≥ 5× speedup on FRFR-category with default settings (MLE + 1000 bootstraps + 5 restarts), wall-clock < 120 seconds on CI hardware.
- **FR-031**: A benchmark script at `scripts/benchmark_fit.py` MUST run the standard fit command with a fixed seed, record wall-clock time and memory, and emit a one-line machine-readable summary. A CI job MUST run this script and fail the build if wall-clock exceeds the tier gate.
- **FR-032**: Tier 2 (JAX): OPTIONAL. If enabled (config flag or env var), the likelihood + gradient MUST be JIT-compiled via JAX, with vmap over lists and participants. Target: ≥ 20× speedup, wall-clock < 30 seconds. Tier 2 MUST pass all behavioral regression tests and the parameter-recovery test; if it doesn't, it is gated off until fixed. The JAX path MUST support both float64 (default, 1e-10 cross-platform tolerance) and float32 (opt-in via `MS_TCM_JAX_DTYPE=float32` env var or `--jax-dtype float32` CLI flag, 1e-8 tolerance); both dtypes are exercised in CI (Q3).
- **FR-033**: Tier 3 (Rust/C): DEFERRED until Tier 1+2 measurements exist. No Rust or C code enters this feature.
- **FR-034**: Performance optimizations MUST NOT change public API semantics — the same parameter values fed to the same dataset MUST return the same MLE + CI (within FR-023 / Q3 tolerance) before and after each tier.

### Functional Requirements — Paper + documentation

- **FR-040**: `paper/main.tex` §3 (Model) MUST be replaced with the v6 hierarchical CMR formulation. §4 (Derivation of the grouped-bridge equivalence) MUST be updated to the v6 narrative (pre-experimental context + λ reinstatement instead of w_S ≫ w_G reweighting). §4.4 (the v1 numerical anchor) MUST be removed from the body.
- **FR-041**: `paper/main.tex` MUST cite Cornell & Zhang 2025 as the direct theoretical predecessor, with an explicit sentence stating that MS-TCM adds exactly one mechanism (λ, storyline-return reinstatement) on top of their hierarchical CMR.
- **FR-042**: A new subsection (v6 §1.5 equivalent) discussing the identity-vs-embedding choice for M^FC_pre MUST be added to the paper.
- **FR-043**: `paper/CDL-bibliography/cdl.bib` (or local `paper/main.bib` if submodule is read-only) MUST gain an entry for Cornell & Zhang 2025. If the shared submodule is not editable in this feature, the entry goes to a local `paper/local.bib` and `main.tex` includes both bibs.
- **FR-044**: `CLAUDE.md` MUST be updated: the "Project" section MUST point at `notes/two_level_cmr_v6.pdf` as the canonical spec; the parameter-regimes list MUST use v6 symbols (β_enc, β_story, γ_fc, k, λ); the §4.4 numerical anchor guidance MUST be removed or marked historical.
- **FR-045**: `README.md`, `code/README.md`, and `data/README.md` MUST be updated to reflect the v6 model API, parameter names, and the new fit workflow (including Tier 1 performance claim and the benchmark script).
- **FR-046**: `specs/001-ms-tcm-impl/data-model.md`, `specs/001-ms-tcm-impl/quickstart.md`, and all four files under `specs/001-ms-tcm-impl/contracts/` MUST either be updated to the v6 API or superseded by new files under `specs/002-ms-tcm-v6-hcmr/` (with a note in `specs/001-ms-tcm-impl/plan.md` pointing at the successor).
- **FR-047**: A "v6 vs. v1 migration" document MUST be added (either `paper/supplement.tex` §X or `notes/v6_migration.md`) describing which v1 symbols mapped to what v6 symbols, what was retired and why, and how to reinterpret v1 fits retrospectively.
- **FR-048**: Any figure in `paper/figs/` whose content depends on v1 math (e.g. a §4.4-based anchor plot) MUST be regenerated or deleted. The figure PDFs MUST be regenerated from notebooks in `code/notebooks/` per CDL convention — no external image editing.
- **FR-049**: A consistency check script at `scripts/check_paper_consistency.py` MUST verify that zero v1-specific literal strings appear in the compiled paper body, and that `CornZhan25` is both in the bib and cited. The authoritative banned-string and required-string lists live in [contracts/paper-consistency.md](contracts/paper-consistency.md) §2.4 and §2.5; the script MUST consume those lists (not hardcode a subset). At minimum the banned list includes β_G, β_S, w_G, w_S, `\wGret`, `\wSret`, "0.866", "0.80532", "0.806", "w_global + w_storyline", "frozen storyline", and "composite similarity". [A5 resolved: FR-049 now defers to contract §2.4/§2.5 as authoritative]

### Functional Requirements — Repository hygiene

- **FR-060**: The feature branch `002-ms-tcm-v6-hcmr` MUST merge cleanly into `main` after `001-ms-tcm-impl` has merged (or, if `001-ms-tcm-impl` is still in-flight, after rebasing onto it).
- **FR-061**: The committed FRFR-category dataset (`data/raw/frfr_category/`) MUST remain byte-identical — the manifest SHA-256s are unchanged, the reformat script is untouched. The rewrite happens at the model layer only.
- **FR-062**: The `scripts/check_no_duplicate_defs.py` Single-Source-of-Truth check MUST stay green.
- **FR-063**: The cross-platform CI matrix (macOS, Ubuntu, Windows) MUST continue to pass — no platform-specific shortcuts in the new code.
- **FR-064**: No mock objects. No stubs. No silent fallbacks that return fake data. (Global rule — see `CLAUDE.md`.)

### Key Entities

- **Item-level context (`c_item`)**: an N-dimensional unit-norm context vector that drifts at rate β_enc with each encoded item. Synchronizes to `c_story` at event and storyline boundaries.
- **Storyline-level context (`c_story`)**: an N-dimensional unit-norm context vector that drifts at rate β_story only when the current item belongs to its storyline. Reinstated with weight λ on storyline return.
- **Item–context associative matrix (`M^IC`)**: N×N matrix, accumulated as ΔM^IC = c_item(i) · f_i^T during encoding. Drives the retrieval activation a = (M^IC)^T · c_item_cue.
- **Storyline-context associative matrix (`M^SC`)**: N×N matrix, accumulated at storyline switches as ΔM^SC = g_{s(i-1)} · c^story_out^T. Used to read back cached storyline contexts on storyline returns.
- **Pre-experimental matrix (`M^FC_pre`)**: N×N matrix mapping item one-hot vectors to pre-experimental context. Identity by default; optional USE-embedding loader for naturalistic stimuli.
- **ModelParameters (v6)**: fields β_enc, β_story, γ_fc, k, λ (core), plus β_rec and ε_d for free-recall, plus the feature dimensionality N and a seed. Replaces the v1 `ModelParameters` entirely.
- **FitResult (v6)**: unchanged in shape from v1 — per-parameter `{mle, ci_lower, ci_upper, ci_method, n_bootstraps, n_converged}` plus fit-level metadata — but fields reflect the v6 parameter set.
- **Behavioral regression reference curves**: per the Q2-resolved policy (Layer 2), the reference curves are the FRFR-category empirical curves computed from `data/raw/frfr_category/recalled.parquet` and cached as `data/processed/reference_curves/frfr_category_{spc,pfr,lag_crp}.parquet`; Cornell & Zhang 2025 Figure 2 and Kahana et al. 2002 Figure 1 are cited as literature anchors but are not used for numerical comparison.
- **Benchmark record**: a machine-readable one-line summary of `scripts/benchmark_fit.py` runs with fields `tier`, `wall_clock_seconds`, `peak_memory_mb`, `n_bootstraps`, `n_restarts`, `git_sha`, `platform`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On the bundled FRFR-category dataset, the SPC, pFR, and lag-CRP curves simulated from the fitted MS-TCM pass the two-layer tolerance policy (Layer 1 qualitative-shape assertions + Layer 2 ≤ 20 % per-bin relative error against FRFR-category empirical curves, MS-TCM no-worse-than standard-TCM), and the same curves pass under the `--standard-tcm` reduction. Verified by `code/tests/test_behavioral_regression.py`.
- **SC-002**: `ms-tcm fit data/raw/frfr_category --n-bootstraps 1000 --n-restarts 5 --seed 42` completes in under 120 seconds on CI hardware with Tier 1 optimizations (≥ 5× improvement from the current ~600 s baseline, recorded in the 001-feature session notes). Verified by `scripts/benchmark_fit.py` plus CI assertion.
- **SC-003**: If Tier 2 (JAX) is landed, the same command completes in under 30 seconds (≥ 20×), while passing all User Story 1 behavioral regression tests.
- **SC-004**: After this feature, `paper/main.tex` contains zero occurrences of v1-specific literal strings (β_G, β_S, "0.866", "0.80532", "0.806", "frozen storyline", composite-weight sum-to-one language), and one or more explicit citations to `CornZhan25`. Verified by `scripts/check_paper_consistency.py`.
- **SC-005**: After this feature, `CLAUDE.md`, `README.md`, `code/README.md`, `data/README.md`, and any file under `specs/001-ms-tcm-impl/` that references the v1 model is either updated to the v6 API or marked superseded with a forward pointer. Verified by a manual documentation-review checklist filled in before merge.
- **SC-006**: The `--standard-tcm` reduction (λ=0, one storyline) produces MLE + CI identical to standard CMR at 1e-10 cross-platform tolerance on the Tier 1 path (and at 1e-10 float64 / 1e-8 float32 on the Tier 2 JAX path), and passes the behavioral-regression two-layer policy. This is the load-bearing sanity check that the rewrite is correct at the CMR-layer.
- **SC-007**: The parameter-recovery test from the 001 feature (T042) continues to pass on the new model, mapped onto the v6 parameter set (β_enc, β_story, γ_fc, k, λ, β_rec, ε_d).
- **SC-008**: The CDL-bibliography entry for Cornell & Zhang 2025 (Psychological Review, 2025) is present in either `paper/CDL-bibliography/cdl.bib` or a local `paper/local.bib`, with the correct DOI, journal, and author list. A reviewer can click the citation in `main.pdf` and resolve it.
- **SC-009**: The fit wall-clock benchmark is machine-readable and checked in on every commit touching `code/ms_tcm/` (history in `data/processed/benchmarks/benchmark_log.csv`). A regression is surfaced as a CI failure rather than being noticed months later.
- **SC-010**: Zero duplicate function definitions (Constitution II / `scripts/check_no_duplicate_defs.py`); zero mocks; all three platforms (macOS, Ubuntu, Windows) pass in CI.

## Assumptions

- **`notes/two_level_cmr_v6.pdf` is the authoritative spec** for the model equations from 2026-04-22 onward. `notes/ms-tcm.pdf` (v1) remains in the `notes/` directory as historical reference only; it MUST NOT be edited in this feature except to add a "SUPERSEDED BY v6" watermark.
- **FRFR-category dataset remains read-only.** The dataset ships byte-identically; the reformat script is unchanged; the feature encoder output is unchanged.
- **Cornell & Zhang 2025 is a Psychological Review paper** (the preprint in `notes/CornZhan25.pdf` is the basis for this feature). Citation details — authors Charlotte A. Cornell and Qiong Zhang; title "Hierarchical Context Guides Human Memory Search" — will be filled into the bib with the published DOI if available, else with the preprint URL and a `[preprint]` note.
- **Tier 1 is in-scope for this feature.** Tier 2 is in-scope if feasible within this feature's effort budget; otherwise it is a follow-on feature tracked in a new GitHub issue. Tier 3 is always a follow-on.
- **Standard-TCM reduction matters.** The v1 implementation supported a `--standard-tcm` switch; v6 must preserve it, mapping to "λ=0, one storyline, β_list drift only".
- **γ_fc is a free parameter in the fit.** C&Z 2025 fixes it at 0.315 (Table 1). We initialize at that value but fit it, since the FRFR-category multi-hot feature encoder may prefer a different mixture than the one-hot word case.
- **Paper figures that change**: any figure whose generating notebook references v1-only math (e.g. a §4.4 anchor figure) needs regeneration. Existing notebooks that reference only the dataset and the fit output may or may not need regeneration; the notebook-to-figure mapping is verified during execution.
- **Bibliography submodule editability**: if `paper/CDL-bibliography/` is a read-only submodule in this feature's window, a local `paper/local.bib` accommodates the C&Z entry and `main.tex` includes both bibs.
- **Parameter-recovery test scale**: the test remains `@slow` (FRFR-scale synth-and-fit with 40 bootstraps and 2 restarts) and is not run on every commit; a lightweight variant may be added if CI time permits.
- **Benchmark hardware**: "CI hardware" here means the GitHub-hosted runner baseline (2-core, 7-GB RAM on `macos-latest` / `ubuntu-latest`). A local laptop typically runs 2-4× faster.
- **Image resolution guardrail**: any figures regenerated during this feature must be saved at low DPI (≤ 100) when used for visual inspection, per project-level memory of conversation crashes caused by high-DPI image review.
- **Symbol-vs-identifier convention** [A2 resolved]: the mathematical symbol `λ` (spec, paper, `notes/two_level_cmr_v6.pdf` §5) is implemented in Python as the identifier `lambda_reinstate` because `lambda` is a reserved keyword in Python. Similarly, `ε_d` maps to `epsilon_d` and `γ_fc` maps to `gamma_fc`. Tests and paper cite the math symbol; source code uses the identifier.
- **Tier-2 backend-selection precedence** [A8 resolved]: when multiple configuration mechanisms disagree, the precedence is: `--backend` CLI flag > `MS_TCM_BACKEND` environment variable > built-in default (Tier 1). The `[fit] backend` pyproject.toml option mentioned earlier in drafts is DROPPED; env var + CLI flag suffice. Equivalently for `--jax-dtype` / `MS_TCM_JAX_DTYPE`: CLI flag > env var > float64 default.
- **JAX auto-fallback** [A6 resolved]: if `--backend jax` or `MS_TCM_BACKEND=jax` is selected but the `jax` extras group is not installed, the CLI logs a warning ("JAX backend requested but jax is not importable; falling back to Tier 1. Install with `pip install -e .[jax]`.") and continues with the Tier 1 backend. A test (`test_cli.py::test_jax_fallback_when_unavailable`) asserts the warning is emitted and the fit still completes.
- **Branch name**: the feature lives on `002-ms-tcm-v6-hcmr`. Merge target is `main`. If `001-ms-tcm-impl` has not yet merged, this branch rebases onto it before opening a PR.
