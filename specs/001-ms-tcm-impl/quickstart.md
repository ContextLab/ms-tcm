# Quickstart: MS-TCM on FRFR-Category

This quickstart walks a new contributor from a fresh clone to a fully-fit MS-TCM model on the bundled FRFR-category dataset in about five commands. It runs identically on macOS, Ubuntu, and Windows (Constitution Principle IV).

## 0. Prerequisites

- Git
- Python 3.11 (`python --version` should report `3.11.x`)

## 1. Clone and initialize submodules

```bash
git clone <repo-url> ms-tcm
cd ms-tcm
sh setup.sh            # initializes the paper/CDL-bibliography submodule
```

On Windows PowerShell, run the two commands in `setup.sh` by hand: `git submodule init`, `git submodule update`, then `cd paper/CDL-bibliography && git checkout master && cd ../..`.

## 2. Install the package (editable, with dev extras)

```bash
python -m pip install -e ".[dev]"
```

This installs `ms_tcm` from `code/ms_tcm/` and its dependencies (`numpy`, `scipy`, `pyarrow`, `pandas`, `h5py`, `pytest`, `pytest-cov`). Identical on all three platforms.

## 3. Validate the bundled FRFR-category dataset

```bash
ms-tcm validate data/raw/frfr_category
```

Expected: exit code 0 and a one-line summary matching `data/raw/frfr_category/manifest.json` (30 participants, 16 lists/pt, 16 words/list, 4 categories/list, 7680 presented rows, 5215 recalled rows, 393 extra-list intrusions). No network access required — the dataset is committed.

## 4. Fit MS-TCM

```bash
ms-tcm fit data/raw/frfr_category \
  --out data/processed/fits/mstcm \
  --seed 42
```

Expected: completes in under 10 minutes on a current laptop (target < 5 minutes) with the default 1000 bootstrap replicates, 5 random restarts, and 95 % percentile CIs. Outputs:

- `data/processed/fits/mstcm/fit_summary.json` — every parameter's MLE + 95 % CI, plus log-likelihood, AIC, BIC, elapsed seconds, and metadata.
- `data/processed/fits/mstcm/fit_bootstrap.parquet` — every bootstrap draw's parameter estimate.

## 5. Fit the standard-TCM baseline

```bash
ms-tcm fit data/raw/frfr_category \
  --out data/processed/fits/standard_tcm \
  --seed 42 \
  --standard-tcm
```

Expected: a second `fit_summary.json` with `w_storyline.mle == 0.0` (and CI `[0.0, 0.0]`). AIC/BIC should be higher (worse) than the MS-TCM fit under the paper's theoretical argument; a reversal is a reportable scientific result, not a test failure.

## 6. Inspect the fits

```python
import json
mstcm = json.load(open("data/processed/fits/mstcm/fit_summary.json"))
tcm   = json.load(open("data/processed/fits/standard_tcm/fit_summary.json"))

for name, info in mstcm["parameters"].items():
    print(f"{name}: {info['mle']:.3f} "
          f"[{info['ci_lower']:.3f}, {info['ci_upper']:.3f}]")

print("MS-TCM  log L =", mstcm["log_likelihood"], "AIC =", mstcm["aic"])
print("std TCM log L =", tcm  ["log_likelihood"], "AIC =", tcm  ["aic"])
```

The demo notebook at `code/notebooks/demo.ipynb` reproduces Figure 1 from the (eventual) paper using exactly the same API.

## 7. Run the test suite

```bash
pytest code/tests
```

Expected: all tests pass. Highlights:

- `test_composite.py::test_section_4_4_numerical_anchor` — 0.866 grouped / 0.806 bridge to four decimals at β_G = β_S = 0.5, w_G = 0.2, w_S = 0.8, m = 3 (notes/ms-tcm.pdf §4.4).
- `test_model.py::test_single_storyline_matches_standard_tcm` — K = 1 reduction identical within 1e-12.
- `test_frfr.py::test_shipped_dataset_hashes_match_manifest` — the committed data hasn't drifted from `manifest.json`.
- `test_parameter_recovery.py` — synthetic recalls generated from a known MS-TCM parameter vector are recovered by the fitter within the bootstrap 95 % CI (SC-009).

## 8. Optional: regenerate the bundled dataset from the upstream egg

Only needed if you want to verify the adapter end-to-end.

```bash
python scripts/reformat_frfr_category.py
```

The script downloads `exp2.egg` from Dropbox (or uses `--egg <path>` if you already have a local copy), reformats it, and writes `data/raw/frfr_category/`. Running it twice produces byte-identical output (SHA-256 hashes in `manifest.json` unchanged). If the committed dataset's hashes no longer match the output, the repo's committed copy has drifted — rerun the script and commit the difference with a short explanation.

## Troubleshooting

- **`ModuleNotFoundError: ms_tcm`**: re-run step 2; the editable install binds the package to the repo so any subsequent `git pull` is picked up automatically.
- **`pyarrow` install fails**: ensure Python is 3.11. Older Python versions (≤ 3.9) may get an older pyarrow wheel that does not guarantee cross-platform bit-exactness.
- **Manifest hash mismatch on validate**: a file in `data/raw/frfr_category/` was edited outside the adapter. Regenerate via step 8 rather than editing Parquet files by hand.
- **Fit aborts with "fewer than 90 % of bootstraps converged"**: the dataset is too small for the selected optional mechanisms, or `--n-restarts` is too low. Try `--n-restarts 10`; if still failing, drop `--gamma`/`--lambda`.
