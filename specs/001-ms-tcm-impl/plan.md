# Implementation Plan: MS-TCM Model Implementation with FRFR-Category Worked Example

**Branch**: `001-ms-tcm-impl` | **Date**: 2026-04-20 | **Last updated**: 2026-04-21 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-ms-tcm-impl/spec.md`

## Summary

Deliver a Python implementation of MS-TCM (per `notes/ms-tcm.pdf`) together with a bundled real-data worked example — the **category condition (exp2)** of Manning et al. (2023), FRFR — and a maximum-likelihood fitter that reports 95 % participant-level bootstrap confidence intervals for every MS-TCM parameter. The FRFR-category dataset (30 participants × 16 lists × 16 words; 4 semantic categories per list ≡ K = 4 storylines per list) ships in the repository under `data/raw/frfr_category/` as Parquet + CSV + `manifest.json`; a committed `scripts/reformat_frfr_category.py` reproduces those files idempotently from the upstream `.egg`. The fitter exposes a `--standard-tcm` switch (w_S = 0) so the TCM baseline is fit from the same CLI for direct comparison. The §4.4 numerical anchor (0.866 grouped vs 0.806 bridge at β_G = β_S = 0.5, w_G = 0.2, w_S = 0.8, m = 3) is pinned as a hand-derived unit test of the composite-similarity math. Synthetic data generation is deferred to GitHub issue #1; a paper writeup is tracked in issue #2.

## Technical Context

**Language/Version**: Python 3.11 (the current `Dockerfile` pins 3.7 and will be updated — see Complexity Tracking)
**Primary Dependencies**:

- `numpy` (arrays, seeded PRNG via `numpy.random.Generator(PCG64)`)
- `scipy.optimize` (L-BFGS-B with random restarts for the MLE fitter)
- `pyarrow` (Apache Parquet reader/writer — canonical dataset format)
- `pandas` (Parquet round-trip for presented/recalled/bootstrap tables)
- `h5py` (read the upstream FRFR `.egg` once inside the reformat script only)
- `pytest`, `pytest-cov` (tests and coverage)

**Storage**: Local-filesystem dataset directories (`data/raw/frfr_category/` is committed; `data/processed/fits/<name>/` for fit outputs).
**Testing**: `pytest` with deterministic seeds; numerical assertions at 1e-12 absolute tolerance for derived outputs; exact SHA-256 equality for dataset files.
**Target Platform**: macOS, Ubuntu, Windows (Constitution Principle IV).
**Project Type**: Scientific Python library + supporting scripts and notebooks. Library lives at `code/ms_tcm/`; the one-off reformat script lives at `scripts/reformat_frfr_category.py`; notebooks at `code/notebooks/` import from the library. No web/API layer.
**Performance Goals**: `ms-tcm validate` completes in < 10 s on the shipped dataset; `ms-tcm fit data/raw/frfr_category` with 1000 bootstrap replicates completes in < 10 minutes on a current laptop (target < 5 minutes). The §4.4 anchor unit test completes in < 1 s.
**Constraints**: (a) Bit-exact Parquet/CSV/JSON dataset files across macOS/Ubuntu/Windows (SHA-256 manifest); (b) ≤ 1e-12 absolute tolerance on all derived model outputs, including log-likelihoods and MLEs; (c) zero duplicated function definitions (Constitution Principle II); (d) every symbol named to match `notes/ms-tcm.pdf`.
**Scale/Scope**: FRFR-category: 30 × 16 × 16 = 7680 presented rows, 5215 recalled rows (4822 in-list + 393 extra-list intrusions), 15 unique categories across the dataset but exactly 4 per list. Feature-vector dimensionality is the sum of one-hot widths for (category, size, first_letter, length, color_bin, location_bin) — on the order of 40–60 dimensions.

## Constitution Check

Evaluated against the four principles in `.specify/memory/constitution.md` v1.0.0:

| Principle | Gate | Status |
|-|-|-|
| I. Accuracy (NON-NEGOTIABLE) | Every numerical function has a unit test with a hand-derived expected answer; §4.4 anchor pinned to four decimals; MLE fitter is compared against a parameter-recovery test; every test assertion cites a source (notes/ms-tcm.pdf, Manning et al. 2023, or `manifest.json`); no mocks | Pass |
| II. Single Source of Truth | Each function is defined once under `code/ms_tcm/`; notebooks, scripts, CLI import rather than redefine; dataset schema lives in one module (`schema.py`); the reformat script is the one-and-only FRFR adapter | Pass |
| III. Clarity | Public API names mirror `notes/ms-tcm.pdf` symbols (`beta_global`, `beta_storyline`, `w_global`, `w_storyline`, `gamma`, `alpha`, `lambda_interference`); quickstart and data-model docs introduce each symbol before use; fit-summary field names are plain-English (`mle`, `ci_lower`, `ci_upper`, `log_likelihood`) | Pass |
| IV. Reproducibility (NON-NEGOTIABLE) | Reformat script is idempotent and byte-deterministic; SHA-256 manifest verified on load; bootstrap draws seeded via PCG64; the optimizer uses a seeded set of random restarts; cross-platform via `pip install -e .`; Dockerfile updated to Python 3.11 in a dedicated setup task | Pass |

**Post-Phase-1 re-check**: same four principles; the concrete schema and MLE contract below preserve all four. No new violations introduced. Pass.

## Project Structure

### Documentation (this feature)

```text
specs/001-ms-tcm-impl/
├── plan.md              # This file (/speckit.plan output)
├── spec.md              # Feature specification (with 2026-04-20 + 2026-04-21 Clarifications)
├── research.md          # Phase 0 output (/speckit.plan)
├── data-model.md        # Phase 1 output (/speckit.plan)
├── quickstart.md        # Phase 1 output (/speckit.plan)
├── contracts/
│   ├── dataset-schema.md          # Parquet schemas + manifest contract (generalized beyond FRFR)
│   ├── model-api.md               # Python library public surface
│   ├── cli.md                     # CLI entry points (validate / run / fit / reformat)
│   └── fitter.md                  # MLE + bootstrap-CI contract
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
code/
├── ms_tcm/                         # Python package (single source of truth)
│   ├── __init__.py                 # Re-exports public API
│   ├── context.py                  # Global and storyline-specific context updates (§3.2)
│   ├── composite.py                # Composite encoding/retrieval contexts (§3.3–§3.4)
│   ├── similarity.py               # Cosine similarity, softmax recall-probability readout (§3.4)
│   ├── mechanisms.py               # Optional §5 mechanisms (γ, α, λ), all default off
│   ├── features.py                 # Multi-hot feature encoder (category, size, first_letter, ...)
│   ├── model.py                    # MSTCMModel orchestrator + standard_tcm() reduction
│   ├── params.py                   # ModelParameters dataclass + validation
│   ├── schema.py                   # Parquet schemas, manifest contract, validate_dataset
│   ├── dataset.py                  # Dataset dataclass, load_dataset, save_dataset
│   ├── frfr.py                     # FRFR-category loader: Parquet → typed Dataset
│   ├── likelihood.py               # Per-list log-likelihood of observed recalls given params
│   ├── fit.py                      # MLE via L-BFGS-B with random restarts
│   ├── bootstrap.py                # Participant-level bootstrap for 95 % CIs
│   ├── io.py                       # JSON manifest read/write, SHA-256 hashing
│   └── cli.py                      # ms-tcm validate / run / fit entry points
├── notebooks/
│   └── demo.ipynb                  # Reproduces one figure from library imports only
├── tests/
│   ├── test_context.py             # §3.2 updates
│   ├── test_composite.py           # §4.4 anchor: 0.866 / 0.806 to four decimals
│   ├── test_similarity.py          # Cosine/softmax numerics
│   ├── test_model.py               # End-to-end + K=1 TCM reduction + standard-TCM switch
│   ├── test_schema.py              # All seven FR-012 validation rules
│   ├── test_frfr.py                # Load + structural checks against manifest
│   ├── test_likelihood.py          # Per-list likelihood matches a hand-derived 3-word example
│   ├── test_fit.py                 # Optimizer convergence on a small hand-derived case
│   ├── test_bootstrap.py           # Bootstrap determinism + CI ordering
│   └── test_parameter_recovery.py  # SC-009: synth-and-fit recovers params within 2 bootstrap SEs
├── scripts/
│   └── reformat_frfr_category.py   # One-shot egg → Parquet/CSV/manifest reformatter (COMMITTED)
├── data/raw/frfr_category/         # COMMITTED output of the reformat script
│   ├── presented.parquet
│   ├── recalled.parquet
│   ├── presented.csv
│   ├── recalled.csv
│   └── manifest.json
└── pyproject.toml                  # Packaging: editable install, deps, pytest config
```

**Structure Decision**: **Single-project Python library + committed data + a script alongside.** The project is a scientific simulator and fitter, not a web or mobile app. One installable package (`ms_tcm`) under `code/ms_tcm/` keeps every function in exactly one location (Constitution II). The reformat script lives at `scripts/reformat_frfr_category.py` rather than as a subcommand of `ms-tcm` because it is a one-off adapter for a legacy upstream format (a PyTables-serialized quail egg with Python-2 serialized string attributes) and does not belong on the `ms-tcm` API surface. Its output — already committed under `data/raw/frfr_category/` — is what downstream code actually depends on.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No Constitution violations. Three operational complexities worth calling out (not violations, but they need explicit tasks in `/speckit.tasks`):

| Item | Why needed | Simpler alternative rejected because |
|-|-|-|
| Update `Dockerfile` to Python 3.11 base + drop `brainiak`/`hypertools` pins | `pyarrow` needs Python ≥ 3.9; the template's Python 3.7 base would fail Parquet I/O outright; unused pins add image size | Keeping the template Dockerfile would break FR-015 |
| `scripts/reformat_frfr_category.py` relies on the Python standard-library legacy serialization reader for ~3 fields | The upstream egg stores a few Python-2 `newstr` string attrs using a legacy in-band serialization; there is no alternative representation available without re-running the original experiment. Use is scoped to decoding a fixed set of string-valued HDF5 attributes and is documented in the script's module docstring. | Ignoring the affected fields loses category/word identity; shipping a pre-decoded CSV instead of the Parquet+CSV pair breaks the binary/text symmetry used for fast numerics |
| `code/tests/` parallel to `code/ms_tcm/` rather than top-level `tests/` | Keeps all implementation artifacts under `code/` per CDL template convention; matches `code/notebooks/` layout | A top-level `tests/` would break the CDL convention and force a second packaging root |

## Notes

- The synthetic-dataset work described in Session 2026-04-20 clarifications (Parquet-by-seed generator, per-storyline cluster centers, bit-exact cross-platform Parquet, default scale K=4 × 30 events × m∈{2…6}) is **deferred to [GitHub issue #1](https://github.com/ContextLab/ms-tcm/issues/1)**. The design decisions recorded in the spec's Clarifications section are preserved so that issue can be executed without re-deciding them.
- The eventual paper writeup that will report the fits produced by this feature is tracked in **[GitHub issue #2](https://github.com/ContextLab/ms-tcm/issues/2)**.
- The FRFR upstream `.egg` hosted on Dropbox is not committed. Its SHA-256 is recorded in `data/raw/frfr_category/manifest.json` so any reviewer who re-downloads and re-reformats can verify parity with the shipped Parquet/CSV.
