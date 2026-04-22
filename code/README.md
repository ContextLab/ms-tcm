# code/

- `ms_tcm/` — the Python package (`pip install -e .` from the repo root).
  Public API: `MSTCMModel`, `ModelParameters`, `Dataset`, `load_frfr_category`,
  `fit_mle`, `bootstrap_ci`, `validate_dataset`, `encode_features`,
  `list_log_likelihood`, `dataset_log_likelihood`, `sample_recalls`.
  See [specs/001-ms-tcm-impl/contracts/model-api.md](../specs/001-ms-tcm-impl/contracts/model-api.md).
- `analyses/` — behavioural-measure computations shared by notebooks and
  figure scripts: serial-position curve, probability of first recall, lag-CRP,
  temporal/category/semantic clustering, sentence-transformer embedding
  loader, and the synthetic-dataset drawing helpers used to build
  posterior-predictive bands from a saved fit.
- `figures/` — one script per paper figure (Constitution II: 1:1 mapping).
  - `make_fig_model.py` → `paper/figs/source/fig_model.pdf`
  - `make_fig_experiment.py` → `paper/figs/source/fig_experiment.pdf`
  - `make_fig_analyses.py` → `paper/figs/source/fig_analyses.pdf` (requires
    `data/processed/fits/mstcm/fit_summary.json` for the predicted bands)
  - `make_fig_clustering.py` → `paper/figs/source/fig_clustering.pdf`
    (requires the embedding parquet from `scripts/compute_embeddings.py`)
  - `make_param_table.py` → `paper/figs/source/param_table.tex`
- `scripts/` — one-off utility scripts (e.g. `compute_embeddings.py`).
- `tests/` — pytest suite. Run `pytest code/tests`. Slow tests
  (parameter-recovery) are excluded from the default run; enable with
  `pytest -m "slow or not slow"`.
- `notebooks/demo.ipynb` — MS-TCM vs. standard-TCM comparison on the bundled
  FRFR-category dataset; imports from `ms_tcm` rather than redefining any
  functions.

Every function is defined exactly once in `ms_tcm/` (and its sibling
`analyses/`); notebooks import rather than redefine. The
`scripts/check_no_duplicate_defs.py` gate enforces this.

## Typical workflow

```bash
pip install -e ".[dev]"                    # install package
ms-tcm validate data/raw/frfr_category     # verify bundled dataset
python code/scripts/compute_embeddings.py  # semantic embeddings (one-shot)
ms-tcm fit data/raw/frfr_category --out data/processed/fits/mstcm \
    --seed 42 --n-bootstraps 40 --n-restarts 2
ms-tcm fit data/raw/frfr_category --out data/processed/fits/tcm \
    --seed 42 --n-bootstraps 40 --n-restarts 2 --standard-tcm
python code/figures/make_fig_model.py      --force-rerun
python code/figures/make_fig_experiment.py --force-rerun
python code/figures/make_fig_analyses.py   --force-rerun
python code/figures/make_fig_clustering.py --force-rerun
python code/figures/make_param_table.py    --force-rerun
(cd paper && ./compile.sh)                 # produces paper/main.pdf
```

Or run everything end-to-end:

```bash
./reproduce.sh
```
