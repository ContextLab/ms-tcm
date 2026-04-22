# Feature Specification: MS-TCM Model Implementation with FRFR-Category Worked Example

**Feature Branch**: `001-ms-tcm-impl`
**Created**: 2026-04-20
**Last updated**: 2026-04-21
**Status**: Draft
**Input**: User description: "now we need to implement the actual model. base it on the notes (notes/ms-tcm.pdf). we need an appropriate data format, and you should build a synthetic dataset that we can use to test the model." — revised 2026-04-21 to replace the synthetic dataset with the category condition of Manning et al. (2023), bundled locally in `data/raw/frfr_category/`.

## Overview

This feature delivers the first working implementation of the Multi-Stream Temporal Context Model (MS-TCM) described in `notes/ms-tcm.pdf`, together with a reformatted real-data worked example: the **category condition** (exp2) of Manning et al. (2023), [*Feature and order manipulations in a free recall task affect memory for current and future lists*](https://github.com/ContextLab/FRFR-analyses). The MS-TCM implementation covers encoding (global and storyline-specific context vectors with independent drift dynamics), retrieval (task-dependent context reweighting), the composite-similarity readout, and a **maximum-likelihood fitting routine with 95 % bootstrap confidence intervals** for every model parameter. The FRFR-category dataset — 30 participants × 16 lists × 16 words, with four semantic categories per list (K = 4), shipped in the repository as Parquet + CSV under `data/raw/frfr_category/` — serves as the first real-data target: each list's four categories are treated as four parallel storylines, so *early* lists (sorted by category) give grouped-within-storyline structure and *late* lists (randomly ordered) give interleaved structure, mirroring the grouped-vs-bridge contrast that motivates MS-TCM. A standard-TCM reduction (w_S = 0) is run on the same data for model comparison.

## Clarifications

### Session 2026-04-20

- Q: What serialization format should be used for the canonical dataset file? → A: Apache Parquet (columnar)
- Q: How should synthetic event feature vectors be generated? → A: Per-storyline cluster centers with Gaussian jitter
- Q: What is the cross-platform numerical tolerance for reproducibility? → A: Bit-exact for dataset files; ≤1e-12 absolute tolerance for derived model outputs
- Q: What default scale (K, events/storyline, m range, trials/condition) should the synthetic generator use? → A: K=4 storylines, ~30 events/storyline, m ∈ {2…6} for bridge trials, ≥100 cue–target trials per condition
- Q: What default feature-vector dimensionality should the synthetic dataset use? → A: d = 64

### Session 2026-04-21

- Q: Should the worked example be synthetic, or a real dataset? → A: Real. Use the category condition (exp2) of Manning et al. 2023 (FRFR). Each list's four semantic categories map to K = 4 storylines. Early lists (sorted) exercise the grouped contrast; late lists (random) exercise the interleaved/bridge contrast.
- Q: Where does the FRFR dataset live, given the dropbox dependency? → A: Download once via `scripts/reformat_frfr_category.py`, reformat to Parquet + CSV + `manifest.json` under `data/raw/frfr_category/`, commit the reformatted files (≈1 MB total on disk) so downstream analyses need no network access. The script re-runs idempotently and re-verifies SHA-256 hashes.
- Q: What parameter-estimation output is required? → A: Per-parameter maximum-likelihood estimate + 95 % bootstrap confidence interval (resample participants with replacement; ≥1000 bootstrap replicates by default).
- Q: Is a synthetic generator still in scope? → A: No. Removed from this feature; tracked for a possible future revisit in GitHub issue #1. The synthetic per-storyline Gaussian design decisions recorded above (Session 2026-04-20 Q2, Q4, Q5) are preserved in the clarifications log so they are available if the issue is reopened.
- Q: Category-to-storyline mapping? → A: Per-list, K = 4: within each list, the four semantic categories define four storylines. Storyline identity does not carry across lists — each list has its own {S_0, S_1, S_2, S_3}. At list boundaries, both the global and storyline contexts are reset to zero, mirroring the "fresh list" semantics of free recall.
- Q: Should the §4.4 numerical anchor (0.866 grouped / 0.805 bridge) still be tested? → A: Yes, but as a unit test of the model equations (hand-derived, no dataset required), not as a system-level integration test over real data. *Note: notes/ms-tcm.pdf §4.4 prints 0.806 for the bridge case; this is a rounding-cascade artifact (using ρ ≈ 0.866 then ρ⁴ ≈ 0.563 then rounding up at the last step). The closed-form value with ρ = √0.75 is 0.80532…; the test asserts the closed form at 1e-6.*
- Q: What initial value should `c_G(0)` and `c_S(0)` take at each list boundary? (Surfaced during implementation of the §4.4 anchor test: a zero initialization breaks the TCM unit-norm invariant because the drift update ρ·c + β·c^IN preserves norm only when ‖c‖ = 1.) → A: Initialize both to a reserved **list-start unit vector** `e_start` — an orthogonal basis direction (first dimension) that no word feature populates. This matches the canonical Howard & Kahana (2002) TCM convention and makes §4.4's analytical anchor hold from the first encoded word. Feature-vector dimensionality grows from d=70 to d=71 (one extra reserved dimension).

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Simulate cued recall for all four conditions (Priority: P1)

A memory researcher provides MS-TCM parameters and a dataset in the canonical format; the simulator steps through encoding, scores every recall transition, and emits per-trial and per-condition outputs. For hand-derived inputs the simulator reproduces the §4.4 analytical anchor (composite similarity 0.866 grouped vs 0.805 bridge at β_G = β_S = 0.5, w_G = 0.2, w_S = 0.8, m = 3) to four decimals (closed form 0.86603 / 0.80532).

**Why this priority**: without a working forward simulator that reproduces the published numerical anchor under the documented parameter regime, nothing else in the project can be tested or trusted.

**Independent Test**: a unit test runs the composite-similarity computation with hand-constructed unit-norm inputs and the anchor parameters; asserts 0.86603 (grouped) and 0.80532 (bridge) at 1e-6 tolerance. An integration test runs the simulator on a minimal in-memory dataset representing one early and one late list and asserts the expected qualitative ordering.

**Acceptance Scenarios**:

1. **Given** β_G = β_S = 0.5, w_G = 0.2, w_S = 0.8, m = 3, and hand-constructed unit-norm input vectors, **When** the composite-similarity computation is called, **Then** the result equals √0.75 = 0.86603 (grouped) and 0.2·0.5625 + 0.8·√0.75 = 0.80532 (bridge), at 1e-6 tolerance (closed form; notes/ms-tcm.pdf §4.4's printed 0.806 is a rounding-cascade artifact).
2. **Given** a minimal in-memory dataset with one grouped-structure list and one interleaved-structure list sharing the same four categories, **When** the simulator is run with default parameters, **Then** the within-category recall probabilities are materially higher than across-category probabilities in the grouped list, and the across-category probabilities in the grouped vs. interleaved list cluster close under storyline-dominant parameters (w_S ≫ w_G) per §4.3.
3. **Given** any valid parameter set, **When** invalid weights are supplied (w_G + w_S ≠ 1 within 1e-12), **Then** `ValueError` is raised before any computation proceeds.

### User Story 2 — Load and inspect the canonical dataset format (Priority: P1)

A researcher opens the repository's FRFR-category dataset (or any dataset conforming to the schema), reads every field without proprietary tooling, and supplies it to the simulator without any ad-hoc preprocessing. The dataset validation routine detects at least six documented classes of malformation and passes cleanly on the shipped data.

**Why this priority**: the dataset format is the long-lived interface between stimuli, the model, and downstream analysis. Its design stabilizes now so every subsequent notebook, fit, or new dataset (real or simulated) uses the same schema.

**Independent Test**: a validation script reads `data/raw/frfr_category/` and reports zero violations plus a human-readable summary (30 pts, 16 lists/pt, 16 words/list, 4 categories/list, 5215 recalls). Running the same script on a dataset where one field is deliberately corrupted returns non-zero with the offending field named.

**Acceptance Scenarios**:

1. **Given** the shipped FRFR-category dataset, **When** `ms-tcm validate` is run, **Then** it exits 0 and prints a summary matching `data/raw/frfr_category/manifest.json`.
2. **Given** a dataset with a mutated participant field or a manifest whose hashes no longer match the files, **When** the validator runs, **Then** it exits non-zero and names the offending file(s) and row(s).
3. **Given** the FRFR-category dataset, **When** passed to the simulator's encoding routine, **Then** encoding completes with no type-coercion warnings and produces a context-vector trajectory for every presented word.

### User Story 3 — Reformat the raw FRFR data deterministically (Priority: P1)

A reviewer on macOS, Ubuntu, or Windows runs `python scripts/reformat_frfr_category.py`, downloads the source egg (or points at a local copy), produces a Parquet + CSV + manifest set under `data/raw/frfr_category/`, and confirms via SHA-256 that the output matches the version committed to the repository.

**Why this priority**: the repository must not depend on a Dropbox URL to reproduce analyses. This story captures both the initial reformat and the idempotent re-runnability that future reviewers rely on.

**Independent Test**: run the reformat script twice in a row on the same egg. Verify that `manifest.json` is byte-identical across runs and that its recorded hashes match the on-disk Parquet/CSV files.

**Acceptance Scenarios**:

1. **Given** a freshly cloned repository with no network access and the shipped `data/raw/frfr_category/` intact, **When** `ms-tcm validate data/raw/frfr_category` is run, **Then** it exits 0 (manifest hashes verify).
2. **Given** `scripts/reformat_frfr_category.py --egg <path>`, **When** the script is run twice back-to-back on the same input, **Then** both runs produce byte-identical Parquet, CSV, and `manifest.json` outputs.
3. **Given** the script is run with no `--egg` argument, **When** the Dropbox download succeeds, **Then** the output matches the shipped copy and the upstream egg's SHA-256 matches the value already recorded in `manifest.json`.

### User Story 4 — Fit MS-TCM (and standard TCM) with MLE + 95 % bootstrap CIs (Priority: P1)

A researcher issues a single command that fits MS-TCM to the FRFR-category dataset and prints, for each parameter, the maximum-likelihood estimate and a 95 % confidence interval obtained by participant-level bootstrap. The same command, with a `--standard-tcm` flag, fits the baseline model (w_S = 0) on the same data. Model-comparison statistics (log-likelihood, AIC, BIC) for both fits are emitted so MS-TCM can be judged against the baseline.

**Why this priority**: fitting is the scientific payoff. Without MLEs and CIs the project cannot make defensible claims about which parameter regime the data actually support.

**Independent Test**: run `ms-tcm fit data/raw/frfr_category --out data/processed/fits/mstcm` and verify (a) the run completes in under a reasonable wall time on a laptop (<10 minutes for 1000 bootstrap reps; target under 5), (b) every model parameter has a finite MLE and a finite 95 % CI whose lower bound ≤ MLE ≤ upper bound, (c) AIC and BIC are both reported. Re-run with `--standard-tcm` and confirm standard TCM fits more poorly under AIC/BIC than MS-TCM under the storyline-dominant regime.

**Acceptance Scenarios**:

1. **Given** the FRFR-category dataset, **When** `ms-tcm fit <dataset> --out <dir>` is run, **Then** the output directory contains a `fit_summary.json` with per-parameter `{mle, ci_lower, ci_upper}` for every MS-TCM parameter and aggregate statistics (log-likelihood, AIC, BIC, n_bootstraps, elapsed seconds).
2. **Given** a completed MS-TCM fit and a completed standard-TCM fit on the same dataset, **When** the two `fit_summary.json` files are compared, **Then** MS-TCM's AIC and BIC are lower than standard TCM's (expected under the paper's theoretical argument; a failure to hold is a reportable result, not a test failure — the check logs but does not crash).
3. **Given** a fixed seed passed to the fitter, **When** the fitter is run twice on the same dataset, **Then** the MLEs agree to within 1e-8 and the bootstrap CI endpoints agree exactly (same bootstrap draws).
4. **Given** a parameter-recovery test where synthetic recall sequences are generated from MS-TCM with known parameters, **When** the fitter is run on those recall sequences, **Then** each parameter's MLE is within two bootstrap SEs of the true value for ≥ 95 % of parameters (nominal coverage).

### Edge Cases

- **Single storyline / K = 1**: MS-TCM reduces to standard TCM; the simulator MUST produce results matching a pure-TCM reference implementation within 1e-12.
- **Inactive-storyline persistence**: when a storyline is inactive for the remainder of a list, its context remains at its last-active value until list boundary (then resets to zero); no NaN or divide-by-zero.
- **Empty recall list**: participants occasionally fail to recall any item for a list; the simulator MUST treat this as a well-defined outcome (contributes zero recalls to the likelihood, not NaN).
- **Extra-list intrusions**: recalls whose `serial_position` is 0 (per the adapter) do not participate in the lag-CRP likelihood; they are counted in a separate "intrusion rate" diagnostic.
- **List-boundary reset**: both global and storyline contexts reset to zero at each list boundary; the simulator MUST NOT leak context across lists.
- **Hashed manifest mismatch**: the validator MUST reject the dataset when any file's SHA-256 disagrees with the manifest.
- **Fitter numerical failure**: if the optimizer reports non-convergence for a given bootstrap draw, that draw is discarded and flagged; the overall fit does not crash unless fewer than 90 % of draws converge.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The model MUST implement the encoding dynamics from `notes/ms-tcm.pdf` §3.2 exactly: a global temporal context that drifts at every encoding step according to c_G(t) = ρ_G · c_G(t−1) + β_G · c^(IN)_i, and one storyline-specific context per storyline that drifts only when its storyline is active and is frozen otherwise.
- **FR-002**: The model MUST enforce ρ_G = √(1 − β_G²) and ρ_S = √(1 − β_S²) internally so the user supplies only β_G and β_S; passing inconsistent ρ and β values MUST raise an explicit error rather than silently normalizing.
- **FR-003**: The model MUST construct the composite encoding context c_comp(i) = w_G · c_G(t) + w_S · c_S*(t) per §3.3 and enforce w_G + w_S = 1 within 1e-12 (with an explicit error when the constraint is violated).
- **FR-004**: The model MUST implement the retrieval composite c_ret(j) = w_G^ret · c_G(t_j) + w_S^ret · c_S_j(t_j) per §3.4 and allow w_G^ret and w_S^ret to be set independently of the encoding weights so the Experiment 3 task-driven reweighting can be expressed.
- **FR-005**: The model MUST expose the recall-probability readout P(i | j) ∝ sim(c_ret(j), c_comp(i)) · a_i using cosine similarity and an item-specific strength term a_i that defaults to 1 when not supplied; normalization across the candidate set is softmax with temperature 1.
- **FR-006**: The model MUST provide a "standard TCM" reduction accessible via a single parameter switch (setting w_S = 0 and disabling storyline-context updates) so the baseline required for model comparison can be produced from the same code path.
- **FR-007**: The model MUST optionally support the additional mechanisms described in §5 — storyline-resumption reinstatement (γ), conversational-reference reinstatement (α_k), and differential interference (λ) — gated by explicit configuration flags that default to "off".
- **FR-008**: The canonical dataset format MUST be a directory containing Parquet tables plus a plain-text `manifest.json`. Required tables/files for a free-recall dataset (the FRFR-category case and any future dataset of the same shape): `presented.parquet` (one row per presented word, columns `participant`, `list`, `serial_position`, `word`, `category`, `size`, `first_letter`, `word_length`, `color_r`, `color_g`, `color_b`, `pos_x`, `pos_y`, `list_group`) and `recalled.parquet` (one row per recalled word, columns `participant`, `list`, `output_position`, `word`, `category`, `serial_position` (0 = extra-list intrusion), `list_group`).
- **FR-009**: The data format MUST rely only on Apache Parquet for binary tables and JSON/CSV for diff-able sidecars (openly documented, cross-platform via `pyarrow`). Each dataset directory MUST additionally contain `manifest.json` with `source` provenance (paper citation, source URL, source-file SHA-256), `design` metadata (participants, lists/pt, words/list, categories/list, etc.), per-file SHA-256 hashes and byte sizes, and `row_counts`.
- **FR-010**: The repository MUST ship the reformatted FRFR-category dataset under `data/raw/frfr_category/` (total on-disk footprint < 1 MB) so downstream analyses run without network access. A companion CSV rendering of every Parquet table MUST also be committed for diff-ability.
- **FR-011**: The reformat script `scripts/reformat_frfr_category.py` MUST be idempotent, accept either an already-downloaded egg or a network download, and produce byte-for-byte identical output across repeated runs on macOS, Ubuntu, and Windows with the same input egg. Its output MUST match the committed dataset under normal conditions (sha-256 parity against `manifest.json`). **Derived model outputs** (similarities, recall probabilities, log-likelihoods, MLEs) MUST be reproducible across platforms to within an absolute tolerance of **1e-12**.
- **FR-012**: A dataset validation routine MUST read a dataset directory and verify: (1) every required Parquet file is present with the expected columns and types, (2) every presented row has a participant/list/serial-position consistent with the manifest's `design` block, (3) every recalled row with `serial_position > 0` references an existing presented row (same participant, same list, same serial position), (4) `list_group` ∈ {`early`, `late`} and matches `list < 8` vs `list ≥ 8`, (5) there are no duplicate (participant, list, serial_position) rows in presented or (participant, list, output_position) rows in recalled, (6) every file's SHA-256 matches the manifest, (7) the manifest itself is well-formed JSON with the required top-level keys. Violations MUST be reported by file/column/row and the routine MUST exit non-zero on any violation.
- **FR-013**: The implementation MUST ship unit tests that (a) reproduce the §4.4 numerical anchor (0.86603 grouped / 0.80532 bridge, closed form with ρ = √0.75) as a hand-derived check on the composite-similarity math, (b) verify the K = 1 / single-storyline reduction matches standard TCM within 1e-12, (c) verify idempotence of the reformat script, (d) verify the validator catches each of the seven violation classes in FR-012.
- **FR-014**: Every function, constant, and data-schema definition introduced by this feature MUST be defined exactly once and imported elsewhere; duplicate definitions across notebooks or modules are prohibited (Single Source of Truth per constitution v1.0.0).
- **FR-015**: All commands — reformat, validate, simulate, fit — MUST run on macOS, Ubuntu, and Windows via `pip install -e .` plus `ms-tcm <subcommand>` invocation. Platform-specific shell one-liners without equivalents for the other platforms are prohibited.
- **FR-016**: Every name for a quantity in the code (β_G, β_S, w_G, w_S, ρ_G, ρ_S, c_G, c_S, γ, α, λ) MUST match `notes/ms-tcm.pdf` exactly; any deviation MUST be justified in writing and flagged at review time.
- **FR-017**: The model-fitting routine MUST compute a maximum-likelihood estimate for every MS-TCM parameter (β_G, β_S, w_G, w_S, w_G^ret, w_S^ret, γ when enabled, λ when enabled; item-strengths a_i are held at 1 unless a separate feature enables them) by minimizing the negative log-likelihood of the observed recall sequences given the dataset's presented sequences. Optimization MUST use a derivative-free or approximate-gradient method with multiple random restarts (default ≥ 5) and a configurable seed; any optimizer non-convergence MUST be reported in the fit summary rather than silently substituted.
- **FR-018**: The model-fitting routine MUST report a **95 % confidence interval** for every fit parameter via participant-level bootstrap (resample participants with replacement, refit per resample, default ≥ 1000 resamples; the bootstrap draws are fully determined by the user-supplied seed so CIs are reproducible). Bootstrap iterations that fail to converge are discarded and flagged; the overall fit aborts if fewer than 90 % converge.
- **FR-019**: The fitter MUST write a `fit_summary.json` containing, for every parameter: `{mle, ci_lower, ci_upper, ci_method, n_bootstraps, n_converged}`, plus fit-level metadata (`log_likelihood`, `aic`, `bic`, `n_participants`, `n_lists`, `n_recalls_used`, `seed`, `elapsed_seconds`, `ms_tcm_version`, `dataset_manifest_sha256`). A companion `fit_bootstrap.parquet` records every bootstrap draw's per-parameter estimate for downstream analyses.
- **FR-020**: The fitter MUST support a `--standard-tcm` switch that constrains w_S = 0 (and disables storyline-context updates) so the baseline model is fit to the same data from the same CLI with no code changes; the two fits MUST be directly comparable (same likelihood definition, same participant pool, same bootstrap seed).
- **FR-021**: For MLE/CI validation, the test suite MUST include a parameter-recovery test: synthesize recall sequences from MS-TCM with known parameters, fit, and assert that each parameter's MLE is within two bootstrap SEs of its true value for at least 95 % of parameters (nominal coverage).

### Key Entities *(include if feature involves data)*

- **Participant**: one human participant in FRFR (or equivalent future datasets). Attributes: participant id (dense integer within a dataset). Downstream analyses group bootstrap resampling at this level.
- **List**: one study-then-recall unit. Attributes: list id (0..15 within a participant), `list_group` ∈ {early, late} derived from `list < 8`. Context vectors reset at each list boundary.
- **Presented word** (event): one studied item. Attributes: participant, list, serial_position (1-based within list), word, category, size, first_letter, word_length, color (r, g, b), position (x, y), list_group. `category` is the storyline label within its list (K = 4 unique categories per list).
- **Recalled word**: one recall output. Attributes: participant, list, output_position (1-based within list's recall period), word, category, serial_position (1..16 if from the list; 0 if extra-list intrusion), list_group.
- **Model parameters**: the numerical knobs governing a single MS-TCM (or standard-TCM) fit. Attributes: β_G, β_S, w_G, w_S, w_G^ret, w_S^ret, γ, α_enabled, λ, feature_dim (derived from the dataset's one-hot-over-features encoding), seed. Enforced constraints: sum-to-1 on encoding and retrieval weights within 1e-12; 0 < β < 1; non-negative γ, λ.
- **Fit result**: the output of one fitter invocation. Attributes: per-parameter {mle, ci_lower, ci_upper, ci_method, n_bootstraps, n_converged}; log_likelihood; AIC; BIC; elapsed_seconds; dataset_manifest_sha256; the `fit_bootstrap.parquet` sidecar with per-draw estimates.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A researcher can, on a freshly cloned repository, run `pip install -e .` followed by `ms-tcm validate data/raw/frfr_category` and see exit code 0 within 10 seconds on macOS, Ubuntu, and Windows with no network access. The 10-second bound is asserted in `test_cli.py::test_validate_completes_under_10_seconds` on CI.
- **SC-002**: The unit test for the §4.4 numerical anchor asserts composite similarity equals the closed-form values 0.86603 (grouped) and 0.80532 (bridge) at 1e-6 tolerance, given β_G = β_S = 0.5, w_G = 0.2, w_S = 0.8, m = 3. These reduce (to three decimals) to the 0.866 and 0.806 values printed in `notes/ms-tcm.pdf` §4.4; the 0.806 is a rounding-cascade artifact of the PDF's worked example using the truncated ρ = 0.866 instead of the exact √0.75.
- **SC-003**: Re-running `python scripts/reformat_frfr_category.py --egg <same egg>` twice produces byte-identical Parquet, CSV, and `manifest.json` outputs (SHA-256 hashes unchanged).
- **SC-004**: Running `ms-tcm fit data/raw/frfr_category` with the default 1000 bootstrap replicates completes in under 10 minutes on a current laptop and emits a `fit_summary.json` with finite MLEs and finite 95 % CIs for every parameter. CI runs a reduced version (`--n-bootstraps 50`) and asserts `fit_summary.json:elapsed_seconds < 60` as a canary; the full-scale 10-minute bound is verified manually and recorded in the paper.
- **SC-005**: The dataset validator catches each of the seven documented malformation classes in FR-012 and passes cleanly on the shipped FRFR-category dataset.
- **SC-006**: Standard TCM (via `--standard-tcm`) fit to the same FRFR-category data is **compared** against MS-TCM's fit by AIC and BIC. This is a **scientific expectation (MS-TCM preferred under the paper's theoretical argument), not a CI gate.** The CI test asserts only that both fits complete, that their AIC/BIC values are finite, and that the comparison is recorded in `fit_summary.json`. The direction of the comparison is reported in the paper (issue #2) but does not fail the build.
- **SC-007**: Every function, constant, and data-schema definition introduced by this feature is defined exactly once and referenced by all other code; a repository-wide check finds zero duplicated definitions.
- **SC-008**: Every numerical assertion in the project's tests traces back to a specific equation, section, or table in `notes/ms-tcm.pdf`, in Manning et al. (2023), or in `data/raw/frfr_category/manifest.json`; a reviewer can verify each assertion in under one minute.
- **SC-009**: Parameter recovery: when recall sequences are synthesized from known MS-TCM parameters and re-fit, at least 95 % of parameters fall within two bootstrap SEs of the true value (nominal coverage for a 95 % CI).

## Assumptions

- `notes/ms-tcm.pdf` is the authoritative source for model equations, symbols, and the §4.4 numerical anchor; wherever the spec and legacy TCM literature conflict, the spec wins.
- The Manning et al. (2023) FRFR category condition (exp2) provides the worked example. Its raw source is a single 45 MB `exp2.egg` HDF5 file hosted on Dropbox; the repository carries the reformatted version (< 1 MB Parquet + CSV + JSON manifest) so downstream work is network-free.
- Per-list storyline mapping (K = 4 per list, categories reset at list boundaries) is the canonical interpretation. Global cross-list drift is not modeled in this feature; cross-list effects (the paper's "future list" contribution) are out of scope and will be addressed in a follow-on feature.
- Feature vectors c^IN_i for MS-TCM are multi-hot encodings of each word's paper-documented features (category, size, first letter, length, color bin, location bin). Exact dimensionality is determined by the one-hot scheme chosen in the plan; the model does not assume a fixed d a priori, it reads it from the dataset.
- Dataset files are required to be **bit-exact** across macOS, Ubuntu, and Windows (hash-matching `manifest.json` entries). Derived model outputs (similarities, recall probabilities, log-likelihoods, MLEs) are required to match within **1e-12 absolute tolerance** across platforms.
- The 95 % confidence intervals are **participant-level percentile bootstraps** by default. More elaborate CI methods (bias-corrected accelerated, profile likelihood) are acceptable enhancements but not required.
- The synthetic data generator (Session 2026-04-20 clarifications) is deferred to GitHub issue [#1](https://github.com/ContextLab/ms-tcm/issues/1). The eventual paper tying MS-TCM to these fits is tracked in GitHub issue [#2](https://github.com/ContextLab/ms-tcm/issues/2).
- The `paper/CDL-bibliography` submodule is already initialized (`setup.sh`) before any notebooks that reference the bibliography are expected to render.
