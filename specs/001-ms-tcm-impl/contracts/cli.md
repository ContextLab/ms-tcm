# Contract: Command-Line Interface

**Producer**: `ms_tcm.cli`
**Installed name**: `ms-tcm` (via `[project.scripts]` in `pyproject.toml`)
**Platforms**: macOS (zsh/bash), Ubuntu (bash), Windows (PowerShell, cmd). Identical invocation syntax on all three.

The CLI is a thin wrapper over the Python API in `contracts/model-api.md`. It exists so that researchers can validate datasets and fit the model without opening a Python REPL, which is required to meet SC-001 ("single documented command on a fresh clone on macOS/Ubuntu/Windows").

There is no `ms-tcm generate` subcommand. The FRFR-category dataset is bundled in the repository; regenerating it from the upstream egg is a separate one-shot script (`scripts/reformat_frfr_category.py`) that lives outside the `ms-tcm` CLI because it adapts a legacy upstream format and does not belong on the stable API surface.

## 1. `ms-tcm validate`

Validate a dataset directory against the seven rules documented in `data-model.md` §7.

```text
ms-tcm validate <dataset-dir> [--json]
```

- Reads every Parquet and CSV file; re-hashes each one against `manifest.json`.
- Default output is human-readable; `--json` emits machine-readable output suitable for CI.
- Exit code 0 iff `ValidationReport.ok == True`.

## 2. `ms-tcm fit`

Fit MS-TCM (or standard TCM) to a dataset; report per-parameter MLE + 95 % bootstrap CI.

```text
ms-tcm fit <dataset-dir> --out <fit-dir>
          [--seed N]               # default: 0; splits into restart + bootstrap streams
          [--n-bootstraps N]       # default: 1000
          [--n-restarts N]         # default: 5
          [--ci FRAC]              # default: 0.95
          [--standard-tcm]         # constrain w_storyline = 0 for baseline
          [--gamma]                # enable §5.1 resumption reinstatement
          [--alpha]                # enable §5.2 conversational references (requires edge table)
          [--lambda]               # enable §5.3 differential interference
          [--no-csv]               # skip writing CSV mirror of fit_bootstrap.parquet
          [--json]                 # print fit_summary to stdout as JSON after writing to disk
```

- `<fit-dir>` is created if missing. If it already exists non-empty and its contents were not produced by `ms-tcm fit`, the tool refuses to overwrite. To force-replace a prior fit, pass `--force`.
- The tool writes `fit_summary.json` (sorted keys, two-space indent) and `fit_bootstrap.parquet` to `<fit-dir>`. `fit_summary.json` contains every required field from `contracts/model-api.md` §9 `FitResult`.
- Invalid flag combinations (e.g., `--standard-tcm` with `--gamma`) raise before any fitting begins.
- Exit code 0 on success; non-zero on any optimizer / bootstrap failure (see §5).

## 3. Cross-platform invariants

- No shell-specific syntax in argument parsing (no `!`, no bash brace expansion). A user can copy-paste the same command across terminals.
- Paths passed to the CLI are resolved via `pathlib.Path` so forward-slash and backslash both work.
- Randomness flows from the `--seed` argument through `numpy.random.Generator(PCG64(seed))`; no reliance on environment variables or time-based entropy.
- Subcommand dispatch is via `argparse` subparsers; `ms-tcm --help` and `ms-tcm fit --help` print usage to stdout and exit 0 on all three platforms.

## 4. Operational contracts

- **Determinism**: `ms-tcm fit <dataset> --seed N` run twice produces byte-identical `fit_summary.json` and `fit_bootstrap.parquet` (sorted JSON, pinned Parquet settings from `dataset-schema.md` §2).
- **Wall-time**: on a current laptop, `ms-tcm fit data/raw/frfr_category` with the default 1000 bootstraps should complete in under 10 minutes (target < 5 minutes). This is measured and reported as `fit_summary.json:elapsed_seconds`.
- **Logging**: every CLI command emits one-line progress updates to stderr. `--json` suppresses human-readable text on stdout but keeps stderr for CI parsing.

## 5. Exit-code conventions

| Code | Meaning |
|-|-|
| 0 | Success |
| 1 | Argument validation error (e.g. bad flag combination, non-existent dataset dir) |
| 2 | Input file missing / unreadable |
| 3 | Schema validation failed (`validate_dataset` returned `ok=False`) |
| 4 | Fit aborted: fewer than 90 % of bootstrap draws converged |
| 5 | Unexpected runtime error (raised exception; traceback written to stderr) |

## 6. Future subcommands (out of scope for this feature)

- `ms-tcm simulate` (forward-run MS-TCM without fitting; currently available only via the Python API) — tracked informally; no issue yet.
- `ms-tcm generate` (synthetic dataset generator) — tracked in GitHub issue #1.
