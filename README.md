# MS-TCM: Multi-Stream Temporal Context Model

This repository implements the Multi-Stream Temporal Context Model (MS-TCM) — an extension of TCM (Howard & Kahana, 2002) that adds storyline-specific context vectors which drift only during encoding of events from their own storyline. The model is described in [notes/ms-tcm.pdf](notes/ms-tcm.pdf); the first worked example fits MS-TCM to the category condition of Manning et al. (2023) [Feature and order manipulations in a free recall task](https://github.com/ContextLab/FRFR-analyses).

## Quickstart

```bash
git clone <this repo>
cd ms-tcm
sh setup.sh                                 # initializes CDL-bibliography submodule
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                     # installs the ms_tcm package
ms-tcm validate data/raw/frfr_category      # sanity-check the bundled dataset
ms-tcm fit data/raw/frfr_category --out data/processed/fits/mstcm --seed 42
ms-tcm fit data/raw/frfr_category --out data/processed/fits/tcm --seed 42 --standard-tcm
pytest code/tests                            # run the full test suite
```

See [specs/001-ms-tcm-impl/quickstart.md](specs/001-ms-tcm-impl/quickstart.md) for more detail.

## What's in the repo

```
code/ms_tcm/      Python package (model, dataset format, likelihood, MLE, bootstrap CIs, CLI)
code/tests/       unit + integration tests (43 passing)
code/notebooks/   demo notebook
data/raw/frfr_category/    FRFR category-condition dataset (30 pts × 16 lists × 16 words)
paper/            LaTeX source + CDL-bibliography submodule
scripts/          one-off adapters (e.g. FRFR egg -> Parquet reformat)
notes/            ms-tcm.pdf — the canonical model specification
specs/            Spec Kit artifacts: spec, plan, tasks, data-model, contracts
```

## Tracked work

- [Issue #1](https://github.com/ContextLab/ms-tcm/issues/1): synthetic multi-stream dataset generator (deferred from feature 001)
- [Issue #2](https://github.com/ContextLab/ms-tcm/issues/2): paper writeup of MS-TCM + FRFR fits

## License

See [LICENSE](LICENSE).
