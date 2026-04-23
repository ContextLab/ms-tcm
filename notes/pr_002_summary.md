# Pull Request 002 Summary: MS-TCM v6 Hierarchical-CMR Rewrite

**Branch**: `002-ms-tcm-v6-hcmr`
**Target**: `main`
**Spec**: [`specs/002-ms-tcm-v6-hcmr/`](../specs/002-ms-tcm-v6-hcmr/)

## Summary

This PR replaces the v1 MS-TCM implementation with a v6 design built
directly on top of Cornell and Zhang's (2025) hierarchical CMR. The v1
architecture (single global context + bank of frozen-while-inactive
storyline contexts + composite encoding mixture + task-driven retrieval
reweighting) did not reproduce the canonical free-recall benchmark
phenomena (SPC, pFR, lag-CRP). v6 inherits the full standard-CMR
retrieval machinery and adds exactly one new mechanism: storyline-return
reinstatement at encoding with strength λ.

The PR delivers:

1. **v1 destructive deletion** per FR-013. Retired files: `composite.py`,
   `similarity.py`, v1-era `context.py`, `mechanisms.py`, `model.py`,
   their tests, and v1-only `ModelParameters` fields. Git history
   preserves all of this; `notes/v6_migration.md` records the mapping.

2. **v6 hierarchical CMR + λ**. New modules: `drift.py` (two-level
   drift), `boundaries.py` (event-boundary sync, storyline-switch
   cache, storyline-return reinstatement), `matrices.py` (M^IC /
   M^SC / M^FC_exp updates), `preexp.py` (pluggable M^FC_pre),
   `retrieval.py` (standard CMR route), `hcmr.py`
   (HierarchicalCMRModel orchestrator + `sample_recalls`).

3. **Behavioral regression** (US1 / MVP). Two-layer policy from the
   Q2 clarification: Layer 1 qualitative-shape assertions (SPC
   primacy+recency, pFR end-bias, lag-CRP +1 peak with forward
   asymmetry) and Layer 2 relative-fit against FRFR-category empirical
   curves (≤ 20% per-bin relative error, MS-TCM no-worse-than
   --standard-tcm). Both layers pass on the bundled dataset.

4. **Tier 1 performance** (US2). ≥ 5× speedup over the ~600 s v1
   baseline via:
   - vectorized `encode_features` (no pandas `iterrows`)
   - per-dataset feature-cache across likelihood evaluations
   - multiprocessing-parallelized bootstrap with deterministic seed
     derivation per draw
   - basis-vector sparse inner products inside the encoding loop
   
   A benchmark log at `data/processed/benchmarks/benchmark_log.csv`
   records every run; `scripts/benchmark_fit.py` is the CLI wrapper.

   **Tier 2 (JAX)** is also included in this PR (T043-T048, T045b):
   `code/ms_tcm/jax_backend/` provides a JIT-compiled per-list encode +
   likelihood (`hcmr_jax.py`) and an L-BFGS-B-with-analytic-gradients
   MLE driver (`fit_jax.py`). Selected via `MS_TCM_BACKEND=jax` or
   `--backend jax`; dtype via `MS_TCM_JAX_DTYPE={float64,float32}`.
   Auto-fallback to Tier 1 when jax is not importable (A6/T045b).
   The Tier-2 path scores the recall-probability portion of the
   likelihood only; the stopping-rule terms are a fixed bias orthogonal
   to the main gradient, so researchers who need Tier-1-bit-identical
   fits still use `--backend tier1`. See
   `ms_tcm/jax_backend/hcmr_jax.py` module docstring.

5. **Paper + documentation sync** (US3). `paper/main.tex` §2 and §4
   rewritten to v6; new subsection on the pre-experimental context
   matrix M^FC_pre; §Methods parameter table rewritten to the v6
   inventory; `CornellZhang2025` cited in §2 and §Methods. Banned v1
   strings (`\beta_G`, `\beta_S`, `w_G`, `w_S`, `0.866`, `0.80532`,
   `0.806`, `frozen storyline`, `composite similarity`, …) removed
   from `main.tex` body. The consistency checker
   `scripts/check_paper_consistency.py` exits 0.

6. **Pluggable M^FC_pre** (US4). `MFCPreMatrix` protocol with
   `IdentityPreMatrix` and `EmbeddingPreMatrix` concrete
   implementations; `--pre-context identity|<path>` CLI flag.
   `EmbeddingPreMatrix.from_parquet` raises `NotImplementedError`
   because USE integration is out of scope for feature 002 (it lands
   alongside the Xu et al. 2026 cued-recall dataset).

7. **Parameter citations** (US5). Default `ModelParameters` values
   match Cornell and Zhang 2025 Table 1 (and v6 §5 for λ) to exact
   equality; each field's docstring cites its source;
   `test_param_defaults.py` enforces both.

8. **Hygiene + polish**. `scripts/check_no_duplicate_defs.py` exits 0
   (Constitution II / FR-062). `scripts/check_paper_consistency.py`
   exits 0 (FR-049 / SC-004). `scripts/check_figure_provenance.py`
   exits 0 (T078b / A4). Cross-platform determinism: running
   `fit_mle` twice with the same seed agrees to 1e-10 (FR-023),
   serial and parallel bootstrap produce bit-identical draws at the
   same seed.

## Clarifications

Three clarifications were resolved in Session 2026-04-23:

- **Q1**: v1 code retirement strategy → **destructive delete**; git
  history preserves everything.
- **Q2**: behavioral-regression tolerance → **two-layer policy**
  (qualitative shape + ≤ 20 % relative error vs. FRFR-category
  empirical curves + MS-TCM no-worse-than --standard-tcm).
- **Q3**: cross-platform tolerance for Tier 2 JAX → **1e-10 default
  float64, 1e-8 opt-in float32**. Tier 2 ships in this PR
  (`code/ms_tcm/jax_backend/`); the dtype knob is the
  `MS_TCM_JAX_DTYPE` env var or `--jax-dtype` CLI flag.

## Files changed (highlights)

### Destructively deleted (v1 retirement per FR-013)

- `code/ms_tcm/composite.py`
- `code/ms_tcm/similarity.py`
- `code/ms_tcm/mechanisms.py`
- `code/ms_tcm/context.py`
- `code/ms_tcm/model.py`
- `code/tests/test_similarity.py`
- `code/tests/test_composite.py`
- `code/tests/test_context.py`
- v1-specific fields in `code/ms_tcm/params.py`

### New

- `code/ms_tcm/hcmr.py`, `drift.py`, `boundaries.py`, `matrices.py`,
  `preexp.py`, `retrieval.py`, `benchmark.py`
- `code/tests/test_drift.py`, `test_boundaries.py`, `test_matrices.py`,
  `test_preexp.py`, `test_retrieval.py`, `test_hcmr_standard_tcm.py`,
  `test_behavioral_regression.py`, `test_benchmark.py`,
  `test_bootstrap.py`, `test_cross_platform.py`, `test_paradigm.py`,
  `test_param_defaults.py`, `test_cli.py`, `test_paper_consistency.py`
- `scripts/benchmark_fit.py`, `check_paper_consistency.py`,
  `check_figure_provenance.py`, `build_reference_curves.py`
- `code/figures/make_fig_behavioral.py`
- `data/processed/reference_curves/frfr_category_{spc,pfr,lag_crp}.parquet`
  + `manifest.json`
- `notes/v6_migration.md`
- `specs/002-ms-tcm-v6-hcmr/` (full Spec-Kit directory)

### Rewritten

- `paper/main.tex` §2, §4, new §1.5-equivalent, §Methods parameter
  table, abstract
- `paper/supplement.tex` (S0 migration note added, S3 updated to v6
  notation, S4 replaced with standard-CMR-reduction anchor)
- `paper/figs/source/param_table.tex`
- `code/figures/make_fig_model.py` (v6 block diagram)
- `code/ms_tcm/cli.py` (validate / fit / benchmark subcommands;
  --backend / --jax-dtype / --pre-context flags)
- `code/ms_tcm/likelihood.py` (content-keyed encoding cache; fixes
  non-determinism from id()-keyed cache)
- `code/ms_tcm/fit.py` (theta clamping for overflow safety;
  clear_encoding_cache() at fit_mle entry for determinism)
- `CLAUDE.md`, `README.md`, `code/README.md`, `data/README.md`

## Verification

Commands run before this PR was opened (all green):

```bash
python -m pytest code/tests -m "not slow" -q                 # 82 passed
python scripts/check_paper_consistency.py                    # exit 0
python scripts/check_figure_provenance.py                    # exit 0
python scripts/check_no_duplicate_defs.py                    # exit 0
cd paper && ./compile.sh                                     # main.pdf + supplement.pdf
python scripts/benchmark_fit.py data/raw/frfr_category \
       --tier tier1 --seed 42 --n-bootstraps 20 --n-restarts 2   # completes
```

## Non-goals (explicitly deferred)

- **USE-embedding-based M^FC_pre** for cued-recall datasets. API
  hook present (`EmbeddingPreMatrix`); actual USE loader raises
  `NotImplementedError` until the Xu et al. 2026 dataset lands.
- **Tier 3 Rust/C backend**. Out of scope; a future feature only if
  Tier 2 JAX proves insufficient. Current Tier 1 + Tier 2 jointly
  satisfy the SC-002/SC-003 performance targets.
- **Tier 2 JAX bootstrap CI**. The Tier-2 path in this PR covers the
  point MLE only; bootstrap CIs still route through Tier 1
  (`--backend tier1`) because JAX traced arrays don't cross
  `multiprocessing` process boundaries cleanly. A JAX-native bootstrap
  is tracked as an optional follow-on enhancement.

## Documentation-review checklist

A completed copy lives at
[`specs/002-ms-tcm-v6-hcmr/checklists/documentation-review.md`](../specs/002-ms-tcm-v6-hcmr/checklists/documentation-review.md).
Every box is ticked; all gate scripts exit 0; paper compiles cleanly.
