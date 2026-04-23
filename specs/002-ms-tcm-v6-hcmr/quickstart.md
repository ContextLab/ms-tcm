# Quickstart: MS-TCM v6

**Feature**: 002-ms-tcm-v6-hcmr
**Audience**: a researcher on a fresh clone who wants to reproduce the paper's main fit, run the behavioral regression tests, and benchmark the fit time.

## Prerequisites

- Python 3.11 or newer.
- Git (for the `paper/CDL-bibliography` submodule).

## 1. Clone and initialize

```bash
git clone <repo-url> ms-tcm
cd ms-tcm
sh setup.sh                                 # initializes the bibliography submodule
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -e ".[dev,figures]"             # Tier 1 path (no JAX)
```

For the Tier 2 JAX path:

```bash
pip install -e ".[dev,figures,jax]"         # adds jax, jaxlib, optax
```

## 2. Validate the bundled dataset

```bash
ms-tcm validate data/raw/frfr_category
```

Expected: exit code 0, summary listing 30 participants × 16 lists × 16 words × 4 categories per list, 5215 recalled rows. Completes in under 10 seconds.

## 3. Compute empirical reference curves (one-time)

```bash
python scripts/build_reference_curves.py data/raw/frfr_category \
    --out data/processed/reference_curves/
```

Produces `frfr_category_spc.parquet`, `frfr_category_pfr.parquet`, `frfr_category_lag_crp.parquet`, and a `manifest.json` with per-file SHA-256. Subsequent invocations are byte-idempotent (Constitution IV).

## 4. Fit MS-TCM on FRFR-category

**Tier 1 (default, < 2 minutes on a modern laptop)**:

```bash
ms-tcm fit data/raw/frfr_category \
    --out data/processed/fits/mstcm_v6 \
    --n-bootstraps 1000 --n-restarts 5 --seed 42
```

**Tier 2 (JAX, < 30 seconds; requires the `jax` extras)**:

```bash
MS_TCM_BACKEND=jax ms-tcm fit data/raw/frfr_category \
    --out data/processed/fits/mstcm_v6_jax \
    --n-bootstraps 1000 --n-restarts 5 --seed 42
```

Opt in to float32 for maximum throughput (loosens cross-platform tolerance from 1e-10 to 1e-8 per Q3):

```bash
MS_TCM_BACKEND=jax MS_TCM_JAX_DTYPE=float32 ms-tcm fit data/raw/frfr_category \
    --out data/processed/fits/mstcm_v6_jax_f32 \
    --n-bootstraps 1000 --n-restarts 5 --seed 42
```

Output: `fit_summary.json` with per-parameter `{mle, ci_lower, ci_upper}` for `beta_enc`, `beta_story`, `gamma_fc`, `k`, `lambda_reinstate`, `beta_rec`, `epsilon_d`, plus `log_likelihood`, `aic`, `bic`, and provenance metadata.

## 5. Fit the standard-CMR baseline

```bash
ms-tcm fit data/raw/frfr_category \
    --out data/processed/fits/standard_cmr \
    --standard-tcm \
    --n-bootstraps 1000 --n-restarts 5 --seed 42
```

The `--standard-tcm` flag sets λ = 0, disables storyline updates, and reduces MS-TCM to Cornell & Zhang 2025's hierarchical CMR on a single storyline. This is the load-bearing sanity check (SC-006): MS-TCM's AIC should not be worse than standard-CMR's AIC; behavioral regression should pass for both.

## 6. Run the behavioral regression test

```bash
pytest code/tests/test_behavioral_regression.py -v
```

Layer 1 asserts qualitative shape (primacy + recency detectable in SPC; pFR weighted to end-of-list; lag-CRP peaks at +1 with forward asymmetry). Layer 2 asserts per-bin relative error ≤ 20 % against the empirical curves from step 3, AND asserts MS-TCM fits no worse than `--standard-tcm` on each of SPC, pFR, lag-CRP. Runs in under 3 minutes.

## 7. Benchmark the fit wall-clock

```bash
python scripts/benchmark_fit.py \
    data/raw/frfr_category \
    --tier tier1 --seed 42 --n-bootstraps 1000 --n-restarts 5
```

Appends one line to `data/processed/benchmarks/benchmark_log.csv` and prints a one-line JSON summary. CI asserts `wall_clock_seconds < 120` for Tier 1.

## 8. Build the paper

```bash
cd paper && ./compile.sh
```

Produces `main.pdf` and `supplement.pdf`. Verify via:

```bash
python scripts/check_paper_consistency.py
```

Exits 0 if the paper body contains no v1 artifacts (β_G, β_S, "0.866", etc.) and does cite Cornell & Zhang 2025.

## 9. Run the full test suite

```bash
pytest code/tests -v
```

Expected: all tests green on macOS, Ubuntu, and Windows. The `@slow` parameter-recovery test and the behavioral regression test push total runtime to ~5 min.

## Troubleshooting

- **"JAX not found" when using `MS_TCM_BACKEND=jax`**: install the extras group: `pip install -e ".[jax]"`. If `jaxlib` fails to wheel-install on Windows, set `MS_TCM_BACKEND=tier1` to fall back to the numpy path.
- **Fit wall-clock exceeds 120 s on Tier 1**: check `os.cpu_count()` — CI hardware (2 cores) may be slower than target. Run on a real laptop, or raise `--n-bootstraps` ceiling (see benchmark log for platform-specific baselines).
- **Behavioral regression fails Layer 2**: inspect which curve failed via the test's error message. Common root causes: bug in `boundaries.py` (event-boundary sync skipped); `beta_rec` or `epsilon_d` at an unrealistic value; `--pre-context` not matching the dataset.
- **Paper build fails with "undefined reference CornellZhang2025"**: add the bib entry to `paper/local.bib` (template in [research.md](research.md) §R5). `paper/main.tex` must include `\bibliography{CDL-bibliography/cdl,local}` (both bibs).
