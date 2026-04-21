# code/

- `ms_tcm/` — the Python package (`pip install -e .` from the repo root).
  Public API: `MSTCMModel`, `ModelParameters`, `Dataset`, `load_frfr_category`,
  `fit_mle`, `bootstrap_ci`, `validate_dataset`, `encode_features`,
  `list_log_likelihood`, `dataset_log_likelihood`, `sample_recalls`.
  See [specs/001-ms-tcm-impl/contracts/model-api.md](../specs/001-ms-tcm-impl/contracts/model-api.md).
- `tests/` — pytest suite. Run `pytest code/tests`.
- `notebooks/` — one notebook per figure in the paper.
  - `demo.ipynb` — MS-TCM vs. standard-TCM comparison on the bundled FRFR-category dataset.

Every function is defined exactly once in `ms_tcm/`; notebooks import rather
than redefine (Constitution Principle II).
