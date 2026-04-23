<!--
Sync Impact Report
==================
Version change: 1.0.0 → 1.1.0
Bump rationale (2026-04-23): Feature 002-ms-tcm-v6-hcmr adopts
`notes/two_level_cmr_v6.pdf` as the canonical model specification, replacing
the v1 `notes/ms-tcm.pdf`. This amendment updates the canonical-spec pointer
throughout the constitution and retires the v1-specific §4.4 numerical
anchor (0.866 / 0.806), replacing it with the v6-native anchor: the
`--standard-tcm` reduction (λ=0, one storyline) MUST pass the Q2 Layer 1
shape-assertion policy for SPC / pFR / lag-CRP. This is a MINOR bump per
§Governance ("expanded guidance within an existing section"): no principle
is added, removed, or weakened; the Accuracy, Single Source of Truth,
Clarity, and Reproducibility principles retain their NON-NEGOTIABLE status.

Principles (unchanged):
- I. Accuracy (NON-NEGOTIABLE) — §4.4 anchor example replaced with v6 anchor
- II. Single Source of Truth
- III. Clarity — symbol list updated to v6 notation
- IV. Reproducibility (NON-NEGOTIABLE)

Sections touched:
- Header paragraph (canonical spec pointer)
- Principle I (§4.4 anchor example replaced)
- Principle III (symbol list updated to v6)
- Additional Constraints > Canonical specification (pointer updated)
- Governance (version footer updated to 1.1.0)

Templates requiring updates: none. The plan-template, spec-template, and
tasks-template carry no hardcoded references to v1 symbols or §4.4.

Historical note: the original 1.0.0 Sync Impact Report is preserved in the
git history at the ratification commit.
-->

# MS-TCM Constitution

The Multi-Stream Temporal Context Model (MS-TCM) project formalizes and tests an
extension of Cornell & Zhang's (2025) hierarchical Context Maintenance and
Retrieval model, adding storyline-return reinstatement at encoding-time (the
λ mechanism). The canonical specification lives in `notes/two_level_cmr_v6.pdf`
(as of version 1.1.0, ratified 2026-04-23); the earlier `notes/ms-tcm.pdf`
(v1) is historical and is superseded — see `notes/v6_migration.md` for the
v1-to-v6 symbol mapping. This constitution governs how the repository's code,
analyses, and writing are produced and maintained.

## Core Principles

### I. Accuracy (NON-NEGOTIABLE)

Every implementation detail, statistic, citation, and claim MUST be independently
verified before it is committed or published. Assumptions are not evidence.

Rules:

- Every function that produces a numerical result MUST have at least one unit
  test that compares its output against a known, hand-derived answer. The
  required numerical-accuracy regression anchor for the MS-TCM core is the
  `--standard-tcm` reduction: when λ=0 and the storyline context is disabled,
  the model MUST reduce to Cornell & Zhang's (2025) hierarchical CMR and
  reproduce the free-recall benchmark phenomena (SPC, p(first recall),
  lag-CRP) on the FRFR-category dataset under the Q2 Layer 1 shape-assertion
  policy defined in `specs/002-ms-tcm-v6-hcmr/contracts/regression-tests.md`.
  The retired v1 §4.4 anchor (0.866 / 0.806) is historical — see
  `notes/v6_migration.md`.
- Every citation MUST be verified by retrieving the cited source
  (web search + fetch + reading the relevant section) and confirming that the
  cited claim appears in it. Citations that cannot be verified MUST be removed
  or replaced.
- Every statistic reported in prose, figures, tables, or notebooks MUST be
  traceable to a specific analysis in `code/notebooks/` that a reviewer can
  re-run to reproduce the same value. Hard-coded or summarized numbers that
  lack an underlying analysis are prohibited.
- Every empirical claim MUST be supported by an actual analysis of actual data.
  Mock objects, placeholder values, and "looks reasonable" estimates MUST NOT
  be used in lieu of real computation; if real functionality is unavailable,
  the code MUST raise an exception rather than fall back silently.
- Rationale: the project's scientific value depends entirely on the trust a
  reader can place in each number and citation. One unverified claim
  contaminates the whole argument.

### II. Single Source of Truth

Every function, constant, schema, and piece of written content MUST have
exactly one authoritative location in the repository. All other uses reference
it; none duplicate it.

Rules:

- Each function, class, or constant MUST be defined in exactly one module. Other
  modules, notebooks, or scripts that need it MUST import it. Copy-pasting
  function bodies across notebooks or scripts is prohibited.
- When a file needs to change, it MUST be modified in place after any
  outstanding work is committed. Parallel copies (`foo_v2.py`, `foo_new.ipynb`,
  `foo_backup.tex`) MUST NOT be checked in; version history is the backup.
- Derived artifacts (figure PDFs, preprocessed data, compiled paper PDFs) MUST
  be regenerable from a single canonical source (a notebook, a preprocessing
  script, or `paper/compile.sh`). If a derived artifact is checked in, the
  generator that produces it MUST also be checked in and MUST reproduce it
  byte-for-byte-equivalent (up to known nondeterminism, which must be documented).
- Shared bibliographic references live in the `paper/CDL-bibliography`
  submodule; local `.bib` files duplicating entries are prohibited.
- Rationale: duplicated definitions drift. A single authoritative definition
  prevents the "fixed in one place, broken in another" failure mode that
  silently invalidates results.

### III. Clarity

All prose, code comments, figure captions, and notebook narrative MUST be
accessible to a reader outside the authors' immediate subfield.

Rules:

- Jargon and formal notation MUST be defined on first use in any
  reader-facing document (paper, supplement, notebook narrative, README).
  Symbols reused from `notes/two_level_cmr_v6.pdf` (β_enc, β_story, γ_fc,
  k, λ, β_rec, ε_d, c^item, c^story, M^IC, M^SC, M^FC_pre, M^FC_exp) MUST
  be introduced with the same meaning given there. The corresponding Python
  identifiers (e.g. `lambda_reinstate` for λ, `gamma_fc` for γ_fc,
  `epsilon_d` for ε_d) substitute reserved keywords and stylistic
  conventions but carry the same semantics. The retired v1 symbols (β_G,
  β_S, w_G, w_S, ρ_G, ρ_S, c_G, c_S, v1 γ resumption) are historical —
  see `notes/v6_migration.md`.
- Claims MUST be proportioned to evidence. Phrases such as "proves",
  "demonstrates conclusively", or "shows that X causes Y" MUST be used only
  when the underlying analysis supports that strength of claim; weaker
  language ("is consistent with", "suggests") is required otherwise.
- Arguments MUST be structured so that each conclusion follows from stated
  premises with no hidden steps. Deductive leaps, unstated assumptions, and
  circular reasoning MUST be corrected when identified.
- Code and notebook names, variable names, and section headings MUST describe
  what the code or section does, not when it was written or who wrote it
  (`compute_composite_similarity` ✅; `new_sim_v3` ✗).
- Rationale: the MS-TCM argument connects memory theory, narrative
  comprehension, and Bayesian statistics. Readers will approach it from
  different directions; clarity is the load-bearing interface.

### IV. Reproducibility (NON-NEGOTIABLE)

All code in this repository MUST run end-to-end on a fresh clone, on any of
macOS, Ubuntu, and Windows, and MUST produce the same results each time it
is executed.

Rules:

- Every analysis, figure, and derived dataset MUST be producible by a
  documented command (a notebook, a script, or `paper/compile.sh`) starting
  from the repository's raw inputs. If execution requires data not in the
  repository, the fetch step MUST be scripted and the source MUST be cited.
- Setup (environment creation, dependency installation, submodule init) MUST
  be driven by scripts that run on macOS, Ubuntu, and Windows. A script that
  assumes one platform MUST have an equivalent counterpart for the others
  (or a cross-platform replacement such as a Docker image or a Python-based
  runner). Platform-specific one-liners that only work on the author's
  laptop are prohibited.
- Setup and preprocessing scripts MUST be idempotent: running them twice MUST
  leave the repository in the same state as running them once, with no
  duplicate files, partial writes, or errors on the second invocation.
- Randomized analyses MUST set and record explicit seeds. The same script with
  the same seed MUST produce the same output across runs on the same platform.
- The Docker image defined by `Dockerfile` is the reference environment.
  Dependencies required by any notebook or script MUST be added to the
  `Dockerfile` (and to any companion requirements manifest) rather than
  installed ad hoc, and the image MUST rebuild successfully from a clean
  cache before dependency changes are merged.
- Rationale: the paper's conclusions are only as strong as a reader's ability
  to re-run the analyses and get the same answers. Platform- or
  environment-dependent results fail this test.

## Additional Constraints

- **Canonical specification**: `notes/two_level_cmr_v6.pdf` is the source of
  truth for model equations, parameter regimes (β_enc, β_story, γ_fc, k, λ,
  β_rec, ε_d), and the four empirical conditions (within-event,
  across-event-within-storyline, across-event-bridge, and the Experiment 2
  condition). `notes/CornZhan25.pdf` is the theoretical predecessor (Cornell
  & Zhang 2025, Psychological Review). `notes/ms-tcm.pdf` (v1) is historical
  only. Discrepancies between code and the v6 spec MUST be resolved by
  updating the spec or the code, not by letting them drift apart.
- **Data discipline**: `data/raw/` is append-only with respect to preprocessing
  (raw inputs are never rewritten by analysis code). All transformations write
  to `data/processed/`. Every file in `data/processed/` MUST be regenerable
  from `data/raw/` via a script or notebook under `code/`.
- **Secrets**: API keys, participant-identifying data, and other credentials
  MUST NOT be committed. `.gitignore` MUST cover any file that may contain
  them, and any accidental commit MUST be rotated and scrubbed from history.
- **Scientific scope**: claims made in the paper MUST be consistent with the
  pilot data and conditions described in `notes/ms-tcm.pdf`. Extensions beyond
  that scope require an explicit extension of the spec.

## Development Workflow

- **Version control**: work is committed frequently with descriptive messages.
  Files are modified in place; backup copies, `*_old.py`, and `*_v2` siblings
  are not checked in. Before rewriting any file, outstanding changes MUST be
  committed so that prior state is recoverable from history.
- **Spec-kit integration**: feature work flows through the `.specify/`
  templates (`specify` → `clarify` → `plan` → `tasks` → `implement`). Each
  `/speckit.plan` output MUST include a Constitution Check that verifies the
  plan satisfies the four Core Principles; violations MUST be recorded in the
  plan's Complexity Tracking table with justification or resolved before
  implementation begins.
- **Review gates before integration**: every change intended for the main
  branch MUST (a) pass all unit tests (including the numerical-accuracy
  anchors required by Principle I), (b) build the paper without errors via
  `paper/compile.sh`, (c) leave every notebook runnable end-to-end, and
  (d) preserve cross-platform setup (Principle IV).
- **Documentation parity**: any change to code, tests, or examples MUST be
  accompanied by the corresponding documentation update (READMEs, notebook
  narrative, paper text) in the same commit or pull request.
- **When checks conflict**: if fixing one check (e.g., a failing test) requires
  touching code that another check (e.g., the paper build) depends on, the
  other checks MUST be re-run after the fix; partial re-verification is not
  sufficient.

## Governance

- **Authority**: this constitution supersedes ad hoc conventions elsewhere in
  the repository. Where `CLAUDE.md`, READMEs, or inline comments conflict with
  the constitution, the constitution wins and the conflicting document MUST be
  updated.
- **Amendment procedure**: amendments are proposed by editing
  `.specify/memory/constitution.md` in a commit whose message explains the
  change and the version bump. Amendments that add or remove a principle, or
  that change the force of a principle (e.g., downgrading a NON-NEGOTIABLE),
  MUST be reviewed by the project lead before merging.
- **Versioning policy**: semantic versioning applies to the constitution.
  - **MAJOR** — backward-incompatible removal or redefinition of a principle
    or governance rule.
  - **MINOR** — addition of a new principle or section, or materially
    expanded guidance within an existing one.
  - **PATCH** — clarifications, wording, typo fixes, or non-semantic
    refinements that do not change what is required.
- **Compliance review**: during `/speckit.plan` and `/speckit.analyze`, each
  plan and each set of generated artifacts is checked against the four Core
  Principles. Unjustified violations block the workflow until resolved.
- **Runtime guidance**: `CLAUDE.md` at the repository root provides tactical
  guidance for automated coding agents and is itself subject to the
  constitution.

**Version**: 1.1.0 | **Ratified**: 2026-04-20 | **Last Amended**: 2026-04-23
