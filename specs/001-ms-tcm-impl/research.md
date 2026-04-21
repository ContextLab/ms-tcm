# Phase 0 Research: MS-TCM + FRFR-Category Worked Example

**Feature**: `001-ms-tcm-impl` — MS-TCM forward simulator, FRFR-category dataset adapter, MLE + bootstrap-CI fitter
**Date**: 2026-04-21 (rewritten from the 2026-04-20 synthetic-dataset framing)

This note records the technology-choice and methodology decisions implicit in `plan.md`'s Technical Context and `spec.md`'s functional requirements. Earlier (2026-04-20) research items that remain relevant — PCG64 seeding, deterministic Parquet, cosine/softmax numerics, packaging — are preserved with minimal edits; new items cover the FRFR adapter, feature encoding, the likelihood, MLE optimization, and the bootstrap procedure.

## R1 — Apache Parquet reader/writer library for Python

- **Decision**: Use `pyarrow` for all Parquet read/write; use `pandas` as a convenience layer when constructing the two `presented` / `recalled` tables. CSV mirrors are written by pandas with default deterministic options.
- **Rationale**: `pyarrow` is the canonical Parquet binding, installs cleanly on macOS/Ubuntu/Windows via `pip`, and gives byte-deterministic output when compression and sorting are controlled. The free-recall schema (scalar-valued columns only — no fixed-size lists) is simple enough that `pandas.DataFrame.to_parquet(engine="pyarrow")` suffices without hand-building Arrow schemas at the table layer.
- **Alternatives considered**: `fastparquet` (less strict about cross-platform determinism); HDF5 via `h5py` (the upstream egg format, rejected at the output layer because it is not as widely adopted for new work and its compression is less deterministic across platforms); plain CSV only (rejected because recall sequences with special characters and large row counts want a typed binary format).

## R2 — Seeded pseudo-random number generator

- **Decision**: Use `numpy.random.Generator(numpy.random.PCG64(seed))` everywhere randomness is required. Split the top-level seed with `numpy.random.SeedSequence(seed).spawn(n)` to obtain independent streams for (a) MLE restart starting points, (b) bootstrap resampling indices, and (c) synthetic-recall sampling. Every helper that needs randomness takes a `Generator` argument; the top-level generator is constructed from a single integer seed supplied by the caller or CLI.
- **Rationale**: `PCG64` is NumPy's recommended modern bit generator, is identical across macOS/Ubuntu/Windows for the same seed, and produces streams independent of NumPy's legacy global `RandomState`. `SeedSequence.spawn` is the NumPy-sanctioned way to derive independent sub-streams from one seed, which eliminates the anti-pattern where two "seeded" helpers unintentionally share a correlated stream. This resolves the ambiguity flagged as A2 during `/speckit.analyze`.
- **Alternatives considered**: Legacy `numpy.random.seed` (weaker cross-version guarantees); Python's `random` (doesn't vectorize); `jax.random` (pulls in JAX for no upside at this scale).

## R3 — Cross-platform bit-exact Parquet and CSV

- **Decision**: Write Parquet with `compression="zstd"`, `compression_level=1`, `use_dictionary=False`, `write_statistics=False`, `version="2.6"`. Sort every table by its primary key before writing. Write CSVs via `pandas.DataFrame.to_csv(..., index=False)` with default line-endings (`\n`). Record SHA-256 hashes of every file in `manifest.json`.
- **Rationale**: zstd at level 1 is deterministic across platforms given the same `pyarrow` version. Dictionary encoding depends on insertion order; Parquet row-group statistics depend on floating-point reductions; disabling both removes the two known sources of cross-platform drift. Sorting before writing removes the last ordering-sensitivity. The manifest lets CI assert bit-exactness on every platform without re-running the adapter.
- **Alternatives considered**: Uncompressed Parquet (simpler but larger); snappy compression (not deterministic across all payloads); assuming Ubuntu-only (fails FR-011).

## R4 — Cosine similarity and softmax readout

- **Decision**: `cosine_similarity(a, b, eps=1e-30) = numpy.dot(a, b) / ((numpy.linalg.norm(a) + eps) * (numpy.linalg.norm(b) + eps))`. Recall probabilities = softmax over candidate scores with temperature 1, using `scipy.special.softmax` for numerical stability (log-sum-exp).
- **Rationale**: These are the most numerically stable single-pass formulations. The epsilon guard prevents NaN when a context vector is exactly zero (an edge case at the first step of a list, before any word has been encoded on the active storyline — the first-recall cue for a storyline that has not been activated). `scipy.special.softmax` applies the standard max-subtraction trick so the softmax is well-behaved even for extreme scores.
- **Alternatives considered**: Raw softmax without log-sum-exp (overflows on large scores); L1 normalization by sum of raw similarities (more sensitive to negative similarities); unit-normalizing context vectors at construction time (incompatible with the §3.2 update rule).

## R5 — Packaging and cross-platform setup

- **Decision**: Ship a `pyproject.toml` at the repo root declaring a single installable package `ms_tcm` located at `code/ms_tcm/`. Dependencies pinned as lower bounds: `numpy>=1.26`, `scipy>=1.11`, `pyarrow>=14`, `pandas>=2.1`, `h5py>=3.10`, `pytest>=7`. Expose one CLI entry point (`ms-tcm`) via `project.scripts` with subcommands `validate` and `fit`. Update the repo's `Dockerfile` to `FROM python:3.11-slim`, `pip install -e .[dev]`. Setup is `python -m pip install -e ".[dev]"` on all three platforms.
- **Rationale**: `pip install -e .` is the universal cross-platform command; it works in bash, zsh, and PowerShell without modification. Pinning lower bounds keeps the install resilient to minor library updates. Exposing the CLI as console scripts gives `ms-tcm validate` and `ms-tcm fit` from any working directory once installed.
- **Alternatives considered**: `conda` / `environment.yml` (the template already uses Docker; adding conda is a third tool); `poetry` (nicer lock files, extra step); separate `setup.py` (legacy).

## R6 — Handling spec edge cases

- **Decision**: Each edge case enumerated in `spec.md` is pinned to a named test under `code/tests/`.
  - K = 1 reduction → `test_model.py::test_single_storyline_matches_standard_tcm`
  - Inactive-storyline frozen context within a list → `test_context.py::test_storyline_context_frozen_when_inactive`
  - List-boundary reset → `test_model.py::test_list_boundary_resets_contexts`
  - Empty recall list → `test_likelihood.py::test_empty_recall_contributes_zero_log_likelihood`
  - Extra-list intrusion skip → `test_likelihood.py::test_intrusions_skipped_in_likelihood`
  - Manifest hash mismatch → `test_schema.py::test_manifest_hash_detects_tampering`
  - Fitter numerical failure → `test_fit.py::test_fit_raises_when_no_restart_converges`
- **Rationale**: Named tests per edge case satisfy Constitution Principle I's "every numerical result has a hand-derived expected answer" rule and make regressions visible in CI.
- **Alternatives considered**: Silent fallbacks — rejected for violating Constitution I.

## R7 — Likelihood and optimizer choice

- **Decision**: Use the dataset log-likelihood defined in `data-model.md` §4.3 as the objective. Minimize NLL with `scipy.optimize.minimize(method="L-BFGS-B")`, reparameterizing all constrained parameters into ℝⁿ so `bounds=None`. Run with `n_restarts=5` from seeded random starts (σ=2 on each logit axis); take the best-converged result.
- **Rationale**: L-BFGS-B is scipy's standard quasi-Newton optimizer, converges robustly for smooth objectives with < 10 dimensions (our case; at most 4–6 free parameters depending on options), and doesn't require an analytic gradient (scipy does finite-difference). The reparameterization (sigmoid / softplus) eliminates boundary-pinning pathologies that afflict bounded optimization on this kind of objective. Multiple random starts mitigate the local-minimum risk that afflicts any non-convex likelihood. CMR-family papers (Polyn et al. 2009; Lohnas et al. 2015) use very similar setups.
- **Alternatives considered**: Nelder–Mead (derivative-free, but slower and less precise in 4–6 dimensions); differential evolution (global, but 10× slower wall-time); analytic gradient via autograd / JAX (premature — add only if L-BFGS-B's finite-difference gradient is too slow on a real fit).

## R8 — Bootstrap CI methodology

- **Decision**: Percentile participant-level bootstrap. For `n_bootstraps=1000`, resample participants with replacement, refit MS-TCM on each resample with its own seeded restart stream (deterministic given the master seed and the bootstrap index), record each draw's parameter vector and convergence flag. 95 % CI endpoints are the 2.5th and 97.5th percentiles of the bootstrap distribution, excluding non-converged draws. Abort the overall fit if < 90 % of draws converge.
- **Rationale**: Participant-level bootstrap is the standard in memory-modeling papers because observations within a participant are not independent. Percentile CIs are the simplest and most widely used method; they don't require symmetry assumptions. The 95 % nominal coverage is checked by the parameter-recovery test (SC-009). 1000 bootstraps is the minimum for reliable 95 % percentile endpoints; more is better but increases wall-time linearly.
- **Alternatives considered**: BCa (bias-corrected accelerated) — more accurate for skewed distributions but requires a jackknife pass; may be added in a follow-on feature. Profile likelihood — more principled but requires re-optimizing at many fixed parameter values; heavier. Normal-approximation CI from the optimizer's Hessian — cheap but unreliable for this non-convex objective.

## R9 — FRFR `.egg` adapter

- **Decision**: The one-shot `scripts/reformat_frfr_category.py` script reads the upstream `exp2.egg` (PyTables HDF5 with a quail-specific group structure), decodes the small set of string-valued Python-2 attribute blobs, and emits `data/raw/frfr_category/` with the two Parquet tables, their CSV mirrors, and `manifest.json`. The upstream egg is not committed; its SHA-256 is recorded in `manifest.json` so any reviewer can verify reproduction.
- **Rationale**: The upstream format is not a general-purpose interchange format and does not belong on the `ms-tcm` public API surface. Pulling the adapter out into a one-shot script keeps the library lean. Committing the reformatted ~1 MB output removes the Dropbox dependency from every downstream workflow (test suite, fit, notebooks, CI).
- **Alternatives considered**: Library subcommand `ms-tcm reformat-frfr` (adds a permanent API surface for a one-shot job); fetching the egg on every CI run (slow, network-dependent); committing the raw egg (45 MB, too large; also a copy of someone else's Dropbox-hosted data with unclear redistribution implications — the reformatted files are derivative works that cite the source).

## R10 — Feature-vector dimensionality and encoder

- **Decision**: Multi-hot encoder with total width d = 70, concatenating category (16), size (2), first letter (26), word-length bin (10), color-r/g/b (4 each), pos_x/pos_y (2 each). Exact bin edges are pinned in `data-model.md` §2 and in the `CATEGORY_BINS`, `LENGTH_BINS`, `COLOR_BIN_WIDTH`, `POSITION_BIN_CUT` constants of `ms_tcm/features.py`.
- **Rationale**: Free-recall TCM-family models traditionally use one-hot or multi-hot word representations for the association matrix `M^TF`; rich semantic embeddings (e.g., Word2Vec) are unnecessary when the paradigm's signal is temporal-contiguity and semantic-clustering along dimensions that are already explicit features of the stimuli. Coarse 4-bin color and 2-bin position keep d small and match the paper's finding that visual features contribute less to recall organization than semantic/lexicographic features.
- **Alternatives considered**: Fine-grained color (256 bins × 3) — inflates d without clear scientific benefit. Continuous features via a concatenation of normalized raw values — raises numerical issues for cosine similarity (length-scale mismatch across dimensions) and requires per-feature normalization. Pre-trained word embeddings (MiniLM, 384 dims) — premature for a worked example and introduces an ML dependency that the fitter doesn't need.
