# MS-TCM: Multi-Stream Temporal Context Model

This repository implements the Multi-Stream Temporal Context Model (MS-TCM)
— an extension of TCM (Howard & Kahana, 2002) that adds storyline-specific
context vectors which drift only during encoding of events from their own
storyline. The model is described in [notes/ms-tcm.pdf](notes/ms-tcm.pdf);
the first worked example fits MS-TCM to the category condition of
Manning et al. (2023), "Feature and order manipulations in a free recall
task affect memory for current and future lists"
([PsyArXiv](https://psyarxiv.com/erzfp),
[code/data](https://github.com/ContextLab/FRFR-analyses)).

## Quickstart

One command reproduces the entire pipeline end-to-end (download → validate →
fit → figures → paper):

```bash
git clone https://github.com/ContextLab/ms-tcm.git
cd ms-tcm
./reproduce.sh
```

Manual quickstart:

```bash
sh setup.sh                                 # init CDL-bibliography submodule
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                     # installs the ms_tcm package
pip install matplotlib sentence-transformers  # figure + embedding extras
ms-tcm validate data/raw/frfr_category      # sanity-check bundled dataset
python code/scripts/compute_embeddings.py   # semantic embeddings (one-shot)
ms-tcm fit data/raw/frfr_category --out data/processed/fits/mstcm \
    --seed 42 --n-bootstraps 40 --n-restarts 2
ms-tcm fit data/raw/frfr_category --out data/processed/fits/tcm \
    --seed 42 --n-bootstraps 40 --n-restarts 2 --standard-tcm
pytest code/tests                           # run the fast test suite
```

See [specs/001-ms-tcm-impl/quickstart.md](specs/001-ms-tcm-impl/quickstart.md)
for more detail.

## What's in the repo

```
code/ms_tcm/      Python package (model, dataset format, likelihood, MLE, bootstrap CIs, CLI)
code/analyses/    behavioural measures (SPC, PFR, lag-CRP, clustering) + synthetic-dataset helpers
code/figures/     one script per paper figure (Constitution II: 1:1 mapping)
code/scripts/     one-off utilities (e.g. sentence-transformer embeddings)
code/tests/       unit + integration tests (45 passing; slow recovery tests opt-in)
code/notebooks/   demo notebook
data/raw/frfr_category/    FRFR category-condition dataset (30 pts × 16 lists × 16 words)
paper/            LaTeX source; main.pdf + supplement.pdf build from Python-generated figures
scripts/          top-level adapters (FRFR egg → Parquet reformat, duplicate-def gate)
notes/            ms-tcm.pdf — the canonical model specification; session notes
specs/            Spec Kit artifacts: spec, plan, tasks, data-model, contracts
reproduce.sh      end-to-end reproducer (venv → validate → fit → figures → paper)
```

## Testing

The default `pytest code/tests` excludes `@slow` tests (parameter-recovery
on synthetic data, each ~10 minutes). To run the full suite:

```bash
pytest code/tests -m "slow or not slow"
```

A separate gate, `python scripts/check_no_duplicate_defs.py`, enforces
Constitution II: every top-level `def` name is defined in exactly one
place across `code/ms_tcm/` and `code/notebooks/`.

## Tracked work

- [Issue #1](https://github.com/ContextLab/ms-tcm/issues/1): synthetic
  multi-stream dataset generator (deferred from feature 001).
- [Issue #2](https://github.com/ContextLab/ms-tcm/issues/2): paper writeup
  of MS-TCM + FRFR fits (drafted in `paper/main.tex`).

## License

See [LICENSE](LICENSE).
