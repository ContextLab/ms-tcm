# v1 → v6 Migration Notes

**Date**: 2026-04-23
**Feature**: [specs/002-ms-tcm-v6-hcmr/](../specs/002-ms-tcm-v6-hcmr/)
**Constitution**: bumped to v1.1.0 in the first commit of this feature.

This document records the v1-to-v6 migration for the MS-TCM project. Anyone reading a v1-era commit, a stale branch, or the retired `notes/ms-tcm.pdf` should consult this file to understand how the symbols, tests, and claims map onto the v6 framework described in `notes/two_level_cmr_v6.pdf`.

## Symbol mapping

The v6 design notes restructure MS-TCM as a direct extension of Cornell & Zhang (2025) hierarchical CMR (see `notes/CornZhan25.pdf`). The parameter inventory changes accordingly.

| v1 symbol (notes/ms-tcm.pdf) | v6 symbol (notes/two_level_cmr_v6.pdf) | Python identifier | Notes |
|-|-|-|-|
| β_G (global drift) | β_enc (item-level drift) | `beta_enc` | Role shifted: v1 β_G drove the single global context; v6 β_enc drives the fast item-level context, with a separate slower β_story for the storyline level |
| β_S (storyline drift) | β_story (storyline-level drift) | `beta_story` | Conceptually preserved; labeled β_list in C&Z 2025 Table 1 |
| w_G (encoding weight) | RETIRED | — | v6 has no composite encoding weight; retrieval is instruction-blind (v6 §3.2) |
| w_S (encoding weight) | RETIRED | — | Same — v6 does not form a composite encoding context |
| w_G^ret, w_S^ret (retrieval reweighting) | RETIRED | — | v1 §3.4 task-dependent reweighting is superseded by v6 §3.2 scoring-filter account: retrieval dynamics are instruction-blind; task instructions act only as a response-scoring filter |
| γ (v1 §5.1 resumption) | λ (storyline reinstatement at encoding) | `lambda_reinstate` | Conceptually related but operates at encoding time on storyline returns (v6 Eq 6), not as an additive retrieval-time boost |
| α_k (v1 §5.2 conversational references) | RETIRED for this feature | — | v6 does not include conversational references; a future cued-recall feature may reintroduce them |
| λ_interference (v1 §5.3) | RETIRED | — | v6 has no release-from-proactive-interference term; retrieval competition is handled by the standard CMR softmax gain `k` |
| ρ_G, ρ_S | ρ_enc, ρ_story | (computed internally) | Each level's ρ = √(1 - β²) is enforced internally so callers supply only β values |
| c_G (global context) | c^item (item-level context) | `c_item` | Renamed; v6 also introduces a second context level `c^story` that v1 did not have (or had only as frozen-when-inactive storyline vectors) |
| c_S (storyline context, frozen) | c^story (storyline-level context, drifts within its own storyline) | `c_story` | Conceptually related; v6 storyline context drifts continuously when its storyline is active and is cached / reinstated at switches and returns via M^SC (not merely frozen) |
| — (new) | γ_fc | `gamma_fc` | Pre- vs experimental context mixture weight (v6 §1.5, Eq 1.5.1); inherited from standard CMR |
| — (new) | k | `k` | Softmax inverse temperature at retrieval; inherited from CMR |
| — (new) | β_rec | `beta_rec` | Within-trial retrieval context drift (free-recall only; inherited from CMR Eq 3) |
| — (new) | ε_d | `epsilon_d` | Stopping-rule rate (free-recall only; inherited from CMR Eq 7) |
| — (new) | M^IC | `M_IC` | Item-context associative matrix (v6 Eq 3), hand-off from standard CMR |
| — (new) | M^SC | `M_SC` | Storyline-context associative matrix (v6 Eq 5), updated at storyline switches |
| — (new) | M^FC_pre, M^FC_exp | `M_FC_pre`, `M_FC_exp` | Pre-experimental and experimentally-accumulated context matrices (v6 §1.5) |

The storyline-return reinstatement with strength λ (v6 Eq 6) is the **sole new mechanism** that MS-TCM introduces over Cornell & Zhang 2025. Everything else is inherited.

The v1 numerical anchor from `notes/ms-tcm.pdf` §4.4 (0.866 grouped / 0.80532 bridge at β_G = β_S = 0.5, w_G = 0.2, w_S = 0.8, m = 3) no longer has a meaning under v6 — there is no composite encoding context and no w_G/w_S mixture. It is retired as a regression anchor. Its v6 replacement is: **the `--standard-tcm` reduction (λ=0, one storyline) MUST reproduce the free-recall benchmark phenomena (SPC, p(first recall), lag-CRP) per the Q2 Layer 1 shape-assertion policy** defined in `specs/002-ms-tcm-v6-hcmr/contracts/regression-tests.md` §2.

## Deletion manifest

The following files and tests were destructively deleted in feature 002-ms-tcm-v6-hcmr per FR-013 (Q1 clarification):

- `code/ms_tcm/composite.py` — v1 encoding/retrieval composite (w_G / w_S mixture). Superseded by v6 retrieval route which has no composite.
- `code/ms_tcm/similarity.py` — v1 cosine-similarity + softmax wrappers. Subsumed into `code/ms_tcm/retrieval.py`.
- `code/ms_tcm/mechanisms.py` — v1 §5 optional mechanisms (γ resumption, α conversational references, λ interference). The v6 λ is a different mechanism and lives in `code/ms_tcm/boundaries.py`.
- `code/ms_tcm/context.py` — v1 context-update helpers. Replaced by `code/ms_tcm/drift.py` with v6 two-level dynamics.
- `code/ms_tcm/model.py` — v1 `MSTCMModel` orchestrator. Replaced by `code/ms_tcm/hcmr.py::HierarchicalCMRModel`.
- `code/tests/test_similarity.py` — v1 cosine/softmax wrapper tests.
- `code/tests/test_composite.py` — contained `test_section_4_4_numerical_anchor` and related w_G+w_S=1 enforcement tests; all v1-specific.
- `code/tests/test_context.py` — v1 drift tests; replaced by `test_drift.py` and `test_boundaries.py`.
- v1-specific fields on `code/ms_tcm/params.py::ModelParameters`: `w_global`, `w_storyline`, `w_global_ret`, `w_storyline_ret`, v1 `gamma` (resumption meaning), `alpha_enabled`, `lambda_interference` (v1 §5.3 meaning), `tau`, `phi_s`, `phi_d`.
- Any notebook cells in `code/notebooks/` referencing the retired symbols (β_G, β_S, w_G, w_S, w_G^ret, w_S^ret).

Git history preserves the deleted content at any commit prior to the Phase-1 retirement commit (`v1 retirement: Phase 1 complete`). A reader who needs to reconstruct a v1 run can `git checkout` the ratification commit of constitution v1.0.0 (see `git log .specify/memory/constitution.md`).

## Rationale

The v1 MS-TCM architecture in `notes/ms-tcm.pdf` — a single global context plus K "frozen when inactive" storyline contexts, combined at encoding into a composite `c_comp = w_G · c_G + w_S · c_S` — was designed to explain the grouped-vs-bridge cued-recall equivalence reported by Xu, Duncan, & Manning (2026). The architecture worked for the equivalence argument at the level of context similarity (v1 §4), but **it did not reproduce the canonical free-recall behavioral phenomena** (serial position curve, probability of first recall, lag-CRP) when fit to the FRFR-category dataset. The root cause, documented in the session notes leading up to feature 002, was architectural: v1 scored recalls via cosine similarity on composite encoding contexts and skipped the standard CMR retrieval route (no `M^IC` associative matrix, no `β_rec` retrieval drift, no `ε_d` stopping rule, primacy and softmax gain bolted on as free parameters over bounded cosine similarities). Without these CMR mechanisms, the model cannot produce temporal contiguity or primacy.

Cornell & Zhang's (2025) hierarchical CMR solves this by inheriting the full CMR retrieval machinery and extending it with a slower list-level context that synchronizes to the item-level context at list boundaries. v6 of the MS-TCM design notes (authored 2026-04-22) rebases MS-TCM on top of Cornell & Zhang 2025: two context levels (item, storyline), boundary synchronization, storyline caching in `M^SC`, full CMR retrieval route. MS-TCM's substantive scientific contribution is then **one** new mechanism: storyline-return reinstatement at encoding with strength λ (v6 Eq 6). This is a cleaner claim than the v1 architecture allowed: MS-TCM is "Cornell & Zhang 2025 plus λ," and the grouped-vs-bridge equivalence is derivable from that single addition under identifiable parameter regimes.

The destructive-delete retirement strategy (Q1 clarification) was chosen over keeping v1 alongside v6 because (a) Git history preserves the v1 code for any reviewer who wants to reconstruct historical runs, (b) keeping both in the active code path invites drift and duplicate-definition violations (Constitution II), and (c) the v1 numerical anchors (§4.4 composite similarity, w_G + w_S = 1 enforcement) would be actively misleading — they suggest precision in a model that doesn't reproduce the benchmark phenomena it's supposed to generalize from.

Going forward: `notes/ms-tcm.pdf` stays in the repository as historical reference. Any new feature references `notes/two_level_cmr_v6.pdf` as the canonical spec. The paper (`paper/main.tex`) is rewritten in feature 002 User Story 3 to match v6 and cite Cornell & Zhang 2025 as the direct theoretical predecessor.

## How to reinterpret v1 fits retrospectively

A v1 `fit_summary.json` (from feature 001) has fields `beta_global`, `beta_storyline`, `w_global`, `w_storyline`, `w_global_ret`, `w_storyline_ret`, `gamma`, `lambda_interference`, `tau`, `phi_s`, `phi_d`. None of these map cleanly onto v6 parameters:

- `beta_global` is NOT β_enc — v1 β_G drove a single global context, whereas v6 β_enc drives a hierarchically-structured item context. Any interpretation of the v1 β_G value as a v6 β_enc must acknowledge that the v1 model had a different architecture.
- `beta_storyline` is NOT β_story — v1 storyline contexts were frozen when inactive; v6 storyline contexts drift, are cached, and are reinstated.
- `w_global`, `w_storyline` — NO v6 analog. The v6 model has no encoding composite.
- `w_global_ret`, `w_storyline_ret` — NO v6 analog. v6 retrieval is instruction-blind.
- `gamma`, `lambda_interference`, `phi_s`, `phi_d` — v1-specific mechanisms with no v6 analog.

The appropriate action for v1 fits is to treat them as historical: they describe the behavior of a specific (non-CMR) architecture at a specific moment in the project's history, and should not be quoted in any new scientific claim. New claims use v6 fits produced under feature 002.
