# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan at
[specs/002-ms-tcm-v6-hcmr/plan.md](specs/002-ms-tcm-v6-hcmr/plan.md).
<!-- SPECKIT END -->

## Project

This repository develops the **Multi-Stream Temporal Context Model (MS-TCM)** — a formal extension of TCM (Howard & Kahana, 2002) that adds storyline-specific context vectors which drift only during encoding of events from their own storyline. The goal is to explain the equivalence of across-event-bridge and across-event-within-storyline cued recall (BF₀₁ = 412) reported in Xu, Duncan, & Manning (2026).

**Canonical spec**: `notes/ms-tcm.pdf` is the source of truth for model equations, parameter regimes (β_G, β_S, w_G, w_S, γ, λ), the four empirical conditions being modeled, and the planned model comparisons (standard TCM vs. MS-TCM vs. "perfect reinstatement" vs. "independent storylines"). Re-read it before making modeling decisions.

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
notes/            project notes, including ms-tcm.pdf (the spec)
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
- **The equivalence is the load-bearing prediction.** Any refactor or parameter change must preserve (or explicitly explain the loss of) the grouped ≈ bridge prediction under the identified regimes: storyline-context dominance (w_S ≫ w_G), task-driven retrieval reweighting in Exp 3, or matched-interference compensation (notes/ms-tcm.pdf §4.3).
- **Numerical sanity check from the spec**: with β_G = β_S = 0.5, m = 3, w_G = 0.2, w_S = 0.8, composite similarity (weighted sum of per-stream cosines) is √0.75 ≈ 0.866 (grouped) and 0.2·0.75² + 0.8·√0.75 ≈ 0.805 (bridge). `notes/ms-tcm.pdf` §4.4 rounds these to 0.866 / 0.806 via a rounding cascade (ρ ≈ 0.866 → ρ⁴ ≈ 0.563), so the closed-form bridge is 0.80532… not 0.806. The unit test at `code/tests/test_composite.py::test_section_4_4_numerical_anchor` asserts the closed form at 1e-6. New code should reproduce these numbers before being trusted.
- **Stimulus representations**: events should be encoded as feature vectors from sentence embeddings of video annotations (spec §8). Don't invent ad hoc feature schemes.
- **Model comparison baseline**: always include standard TCM (single context, β_G only) as the reference; the MS-TCM contribution is measured against it.

## Spec Kit integration

`.specify/` contains Spec Kit templates, workflows, and a constitution stub. The `<!-- SPECKIT START -->` block at the top of this file is read by the Spec Kit workflow — do not remove it. When a plan exists under `.specify/` (via `/speckit.plan` or `/speckit.specify`), consult it for technology choices, project structure, and shell commands before acting.

## Lab-wide reproducibility expectations

- The Docker image is the reproducibility contract for notebooks and the paper build. If a notebook requires a new package, add it to the `Dockerfile` (and, if `pip`, to an explicit requirements list), rebuild, and re-run affected notebooks — don't rely on local `pip install` state.
- `data/raw/` is read-only in spirit. Preprocessing writes to `data/processed/`.
- Figures in the paper must be reproducible from a notebook in `code/notebooks/`. When a figure changes, regenerate its PDF in `paper/figs/` from the notebook rather than editing it externally.
