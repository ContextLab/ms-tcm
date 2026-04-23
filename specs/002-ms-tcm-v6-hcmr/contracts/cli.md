# Contract: `ms-tcm` CLI

**Feature**: 002-ms-tcm-v6-hcmr
**Entry point**: `ms-tcm = "ms_tcm.cli:main"` (`pyproject.toml:[project.scripts]`)

## Global flags

| Flag | Type | Default | Purpose |
|-|-|-|-|
| `--help`, `-h` | bool | — | Show help |
| `--version`, `-V` | bool | — | Print `ms_tcm.__version__` and exit |
| `--backend` | string | `tier1` | One of `tier1`, `jax`. Overridden by `MS_TCM_BACKEND` env var. |
| `--jax-dtype` | string | `float64` | One of `float64`, `float32`. Only meaningful when backend is `jax`. Overridden by `MS_TCM_JAX_DTYPE` env var. |

## Subcommands

### `ms-tcm validate <dataset-dir>`

Unchanged from 001. Validates Parquet + manifest integrity. Exit code 0 on success.

### `ms-tcm run <dataset-dir> --params <params.json>`

Runs the forward simulator (encode-only) and writes per-list trajectories to `--out`. Rarely used outside debugging. Unchanged in shape from 001; underlying model is v6.

### `ms-tcm fit <dataset-dir>`

```text
ms-tcm fit DATASET_DIR [options]

Required:
  DATASET_DIR              Path to a Parquet dataset directory (e.g. data/raw/frfr_category)

Output:
  --out PATH               Directory for fit_summary.json + fit_bootstrap.parquet
                           (required unless --dry-run)

Fit configuration:
  --seed INT               Master seed; default 0
  --n-restarts INT         Number of optimizer restarts; default 5
  --n-bootstraps INT       Number of bootstrap resamples; default 1000
  --paradigm STR           "free_recall" (default) or "cued_recall"

Model configuration:
  --standard-tcm           Reduce MS-TCM to standard CMR (λ=0, one storyline)
  --pre-context STR        "identity" (default) or a path to an embeddings Parquet
  --starting-values PATH   Override C&Z 2025 Table 1 defaults (JSON)

Performance:
  --backend STR            "tier1" or "jax"; default tier1
  --jax-dtype STR          "float64" (default) or "float32"; only used when backend=jax
  --n-processes INT        Bootstrap parallelism; default os.cpu_count()

Logging:
  -v, --verbose            Increase log level
  --dry-run                Parse flags, validate inputs, print plan, do not fit
```

Exit codes:

- 0: fit completed, `fit_summary.json` written.
- 1: non-convergence (> 10 % of bootstraps failed to converge).
- 2: input validation failure (bad dataset, bad params).
- 3: backend unavailable (e.g., `--backend jax` without the extras installed).
- 130: user interrupt.

### `ms-tcm benchmark <dataset-dir>` (NEW in v6)

Runs a deterministic benchmark of the standard fit command and appends one record to `data/processed/benchmarks/benchmark_log.csv`.

```text
ms-tcm benchmark DATASET_DIR [--tier STR] [--seed INT] [--n-bootstraps INT] [--n-restarts INT]
```

Defaults: `--tier tier1 --seed 42 --n-bootstraps 1000 --n-restarts 5`. Emits a one-line JSON summary to stdout and appends a CSV row.

Exit codes: 0 on success; 1 if wall-clock exceeds the tier's gate (Tier 1: 120 s, Tier 2-float64: 30 s, Tier 2-float32: 30 s).

### `ms-tcm list-paradigms` (NEW informational)

Prints the supported paradigms and their required parameter subsets.

## Environment variables

| Variable | Equivalent flag | Purpose |
|-|-|-|
| `MS_TCM_BACKEND` | `--backend` | Override backend without changing command lines |
| `MS_TCM_JAX_DTYPE` | `--jax-dtype` | Override JAX dtype |
| `MS_TCM_N_PROCESSES` | `--n-processes` | Override bootstrap parallelism |

## Examples

```bash
# Default Tier 1 fit (1000 bootstraps, 5 restarts, seed 42)
ms-tcm fit data/raw/frfr_category --out data/processed/fits/mstcm --seed 42

# Standard-CMR baseline on same data
ms-tcm fit data/raw/frfr_category --out data/processed/fits/standard --standard-tcm --seed 42

# Tier 2 JAX fit with float32 for max speed
MS_TCM_BACKEND=jax MS_TCM_JAX_DTYPE=float32 ms-tcm fit data/raw/frfr_category --out data/processed/fits/jax32

# Benchmark
ms-tcm benchmark data/raw/frfr_category --tier tier1
```
