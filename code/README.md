# code/

- `ms_tcm/` — the Python package (`pip install -e .` from the repo root).
  Public API (v6 hierarchical-CMR surface): `HierarchicalCMRModel`,
  `ModelParameters`, `MFCPreMatrix` / `IdentityPreMatrix` / `EmbeddingPreMatrix`,
  `Dataset`, `load_frfr_category`, `fit_mle`, `bootstrap_ci`,
  `benchmark_fit`, `validate_dataset`, `encode_features`,
  `list_log_likelihood`, `dataset_log_likelihood`, `sample_recalls`.
  See [specs/002-ms-tcm-v6-hcmr/contracts/cli.md](../specs/002-ms-tcm-v6-hcmr/contracts/cli.md).
- `analyses/` — behavioural-measure computations shared by notebooks and
  figure scripts: serial-position curve, probability of first recall, lag-CRP,
  temporal/category/semantic clustering, sentence-transformer embedding
  loader, and the synthetic-dataset drawing helpers used to build
  posterior-predictive bands from a saved fit.
- `figures/` — one script per paper figure (Constitution II: 1:1 mapping).
  - `make_fig_model.py` → `paper/figs/source/fig_model.pdf` (v6 architecture)
  - `make_fig_behavioral.py` → `paper/figs/source/fig_behavioral.pdf`
    (SPC / pFR / lag-CRP; FRFR-category empirical vs MS-TCM vs standard-TCM)
  - `make_fig_experiment.py` → `paper/figs/source/fig_experiment.pdf`
  - `make_fig_analyses.py` → `paper/figs/source/fig_analyses.pdf` (requires
    `data/processed/fits/mstcm/fit_summary.json` for the predicted bands)
  - `make_fig_clustering.py` → `paper/figs/source/fig_clustering.pdf`
    (requires the embedding parquet from `scripts/compute_embeddings.py`)
  - `make_param_table.py` → `paper/figs/source/param_table.tex`
- `scripts/` — one-off utility scripts (e.g. `compute_embeddings.py`).
- `tests/` — pytest suite. Run `pytest code/tests -m "not slow"` for the
  fast default; `pytest code/tests -m "slow or not slow"` for the full
  suite including parameter-recovery and the full-FRFR benchmark gate.
- `notebooks/demo.ipynb` — MS-TCM vs. standard-TCM comparison on the bundled
  FRFR-category dataset; imports from `ms_tcm` rather than redefining any
  functions.

Every function is defined exactly once in `ms_tcm/` (and its sibling
`analyses/`); notebooks import rather than redefine. The
`scripts/check_no_duplicate_defs.py` gate enforces this.

## CLI

The `ms-tcm` CLI exposes three subcommands: `validate`, `fit`, `benchmark`.
See [specs/002-ms-tcm-v6-hcmr/contracts/cli.md](../specs/002-ms-tcm-v6-hcmr/contracts/cli.md)
for the full contract.

```bash
ms-tcm validate data/raw/frfr_category
ms-tcm fit data/raw/frfr_category --out data/processed/fits/mstcm \
    --seed 42 --n-bootstraps 1000 --n-restarts 5
ms-tcm fit data/raw/frfr_category --out data/processed/fits/standard \
    --seed 42 --n-bootstraps 1000 --n-restarts 5 --standard-tcm
ms-tcm benchmark data/raw/frfr_category --tier tier1 --seed 42
```

## Typical workflow

```bash
pip install -e ".[dev,figures]"              # install package + figure deps
ms-tcm validate data/raw/frfr_category       # verify bundled dataset
python scripts/build_reference_curves.py data/raw/frfr_category \
    --out data/processed/reference_curves/   # compute empirical SPC/pFR/lag-CRP
ms-tcm fit data/raw/frfr_category --out data/processed/fits/mstcm \
    --seed 42 --n-bootstraps 1000 --n-restarts 5
ms-tcm fit data/raw/frfr_category --out data/processed/fits/standard \
    --seed 42 --n-bootstraps 1000 --n-restarts 5 --standard-tcm
python code/figures/make_fig_model.py        --force-rerun
python code/figures/make_fig_behavioral.py   --force-rerun
python code/figures/make_fig_experiment.py   --force-rerun
python code/figures/make_fig_analyses.py     --force-rerun
python code/figures/make_fig_clustering.py   --force-rerun
(cd paper && ./compile.sh)                   # produces paper/main.pdf
```

Or run everything end-to-end:

```bash
./reproduce.sh
```
