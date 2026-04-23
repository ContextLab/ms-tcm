# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan at
[specs/002-ms-tcm-v6-hcmr/plan.md](specs/002-ms-tcm-v6-hcmr/plan.md).
<!-- SPECKIT END -->

## Project

This repository develops the **Multi-Stream Temporal Context Model (MS-TCM)** — a formal extension of Cornell and Zhang's (2025) hierarchical CMR that adds a single new mechanism: **storyline-return reinstatement at encoding with strength λ** (v6 Eq 6). The goal is to explain the equivalence of across-event-bridge and across-event-within-storyline cued recall (BF₀₁ = 412) reported in Xu, Duncan, & Manning (2026).

**Canonical spec**: `notes/two_level_cmr_v6.pdf` is the source of truth for model equations, the v6 parameter inventory (β_enc, β_story, γ_fc, k, λ, β_rec, ε_d), the four empirical conditions being modeled, and the planned model comparisons (standard CMR via `--standard-tcm` reduction vs. MS-TCM, plus the hierarchical-CMR reduction λ=0). The v1 MS-TCM design (`notes/ms-tcm.pdf`) is retired; see `notes/v6_migration.md` for the v1→v6 symbol mapping. Re-read `notes/two_level_cmr_v6.pdf` before making modeling decisions.

The repository was cloned from the [ContextLab/latex-base](https://github.com/ContextLab/latex-base) template; most of the existing scaffolding (e.g., `paper/main.tex` titled "Template paper", the `brainiak`/`hypertools` Dockerfile, the `trig.pdf` demo figure, template READMEs) is placeholder content from that template and should be replaced with MS-TCM-specific content as the project progresses, not treated as authoritative.

## Repository layout (CDL convention)

```
code/notebooks/   Jupyter notebooks — one per figure in the paper
data/raw/         raw data (not processed)
data/processed/   preprocessed data consumed by notebooks
paper/            LaTeX source; main.tex + supplement.tex compile separately
paper/figs/       figure PDFs; paper/figs/source/ holds source files
paper/CDL-bibliography/   git submodule — shared lab bib (cdl.bib)
.specify/         Spec Kit workflow artifacts (plans, tasks, constitution)
notes/            project notes, including two_level_cmr_v6.pdf (canonical spec), CornZhan25.pdf (theoretical predecessor), v6_migration.md (v1→v6 mapping); ms-tcm.pdf is retained as historical reference only
```

Notebooks are expected to be the canonical way to reproduce each figure — keep the figure/notebook mapping 1:1 and documented in `code/README.md`.

## Initial setup

One-time, after a fresh clone:

```bash
sh setup.sh                    # inits the CDL-bibliography submodule and checks out master
```

`setup.sh` is required because `paper/CDL-bibliography` is a submodule (see `.gitmodules`) and LaTeX compilation will fail without it.

## Common commands

**Compile the paper** (from `paper/`):
```bash
cd paper && ./compile.sh
```
`compile.sh` runs latex → bibtex → latex ×4 → pdflatex for `main`, then the same for `supplement`, then removes aux files. Both `main.pdf` and `supplement.pdf` are produced and committed to the repo. The admin cover letter (`paper/admin/`) has its own `compile.sh`.

**Dockerized environment** (matches lab reproducibility conventions):
```bash
docker build -t cdl .
docker run -it -p 9999:9999 --name cdl -v $PWD:/mnt cdl
# inside container:
jupyter notebook --port=9999 --no-browser --ip=0.0.0.0 --allow-root
```
The Dockerfile currently pins `brainiak`, `hypertools`, `pandas=1.0.5`, etc., inherited from the template. Before relying on it for MS-TCM work, update it to include the dependencies the model actually needs (e.g., sentence embeddings, Bayesian fitting tools) and rebuild.

## Modeling work — conventions specific to this project

- **Four conditions** must be simulated simultaneously when fitting or comparing models: within-event, across-event-within-storyline (grouped, Exp 1), across-event-bridge (interleaved, Exp 3), and the Exp 2 condition. Fitting only one or two conditions misses the target phenomenon.
- **The equivalence is the load-bearing prediction.** Any refactor or parameter change must preserve (or explicitly explain the loss of) the grouped ≈ bridge prediction under the v6 mechanism (storyline-return reinstatement with strength λ; v6 Eq 6, §2.4). Under λ near 1 the bridge target's retrieval cue is driven by the cached storyline context and is nearly independent of the number of intervening other-storyline events; under λ = 0 the model reduces to Cornell and Zhang (2025) hierarchical CMR, which predicts a smaller bridge-grouped gap than standard CMR but not full equivalence.
- **Numerical-accuracy anchor (v6 replacement for the retired v1 §4.4 anchor)**: the `--standard-tcm` reduction (λ=0, single storyline) must pass `code/tests/test_hcmr_standard_tcm.py` and the Layer 1 shape assertions of `code/tests/test_behavioral_regression.py` (SPC primacy+recency, pFR biased to end of list, lag-CRP peak at +1 with forward asymmetry) before any MS-TCM claim is trusted. Both layers cite `notes/CornZhan25.pdf` Figure 2 and the FRFR-category reference curves at `data/processed/reference_curves/`.
- **Stimulus representations**: events should be encoded as feature vectors from sentence embeddings of scene annotations for cued-recall paradigms (v6 §1.5 Option 3, threaded through `MFCPreMatrix`). For the FRFR-category worked example the identity `MFCPreMatrix` is used (v6 §1.5 Option 1). Don't invent ad hoc feature schemes.
- **Model comparison baseline**: always include the `--standard-tcm` reduction (λ=0, single storyline) as the reference; the MS-TCM contribution (storyline-return reinstatement) is measured against it.
- **Performance target**: Tier 1 `ms-tcm fit data/raw/frfr_category --n-bootstraps 1000 --n-restarts 5 --seed 42` completes in under 120 s on CI hardware (FR-031). Use `python scripts/benchmark_fit.py data/raw/frfr_category --tier tier1 --seed 42 --n-bootstraps 1000 --n-restarts 5` to measure; the wall-clock is recorded in `data/processed/benchmarks/benchmark_log.csv`.

## Spec Kit integration

`.specify/` contains Spec Kit templates, workflows, and a constitution stub. The `<!-- SPECKIT START -->` block at the top of this file is read by the Spec Kit workflow — do not remove it. When a plan exists under `.specify/` (via `/speckit.plan` or `/speckit.specify`), consult it for technology choices, project structure, and shell commands before acting.

## Lab-wide reproducibility expectations

- The Docker image is the reproducibility contract for notebooks and the paper build. If a notebook requires a new package, add it to the `Dockerfile` (and, if `pip`, to an explicit requirements list), rebuild, and re-run affected notebooks — don't rely on local `pip install` state.
- `data/raw/` is read-only in spirit. Preprocessing writes to `data/processed/`.
- Figures in the paper must be reproducible from a notebook in `code/notebooks/`. When a figure changes, regenerate its PDF in `paper/figs/` from the notebook rather than editing it externally.
