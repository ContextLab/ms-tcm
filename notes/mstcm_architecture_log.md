# MS-TCM Architecture Iteration Log

This file tracks the evolution of the MS-TCM model architecture across
sessions. Each iteration records the empirical observation that motivated
the change, the proposed mechanism, the rationale, and the predicted
behavioral signatures the change should produce.

The point is to make the reasoning chain reconstructable — when a future
session asks "why does the model have X?", the answer should be findable
here.

---

## Iteration 0 — C&Z 2025 baseline (verified)

**State**: Cornell & Zhang 2025 hierarchical free-recall model
implemented faithfully in `code/ms_tcm/_likelihood_core.py`.

**Verification**: 30 oracle tests + 12 backend-parity tests pass to
1e-10. Reproduces C&Z Fig 2 qualitatively on Kahana 2002 (SPC bowl,
pFR recency-dominant, lag-CRP +1 peak).

**Limit**: single-storyline. No mechanism for category structure.

**FRFR-category fit**: LL = −10699.82, captures SPC + lag-CRP shapes
but cannot produce pFR=1 primacy initiation (structural: first recall
is always sampled from end-of-list cue).

---

## Iteration 1 — MS-TCM with τ-mixture initiation (current, committed)

**Empirical motivation**:
- FRFR-category observed pFR has argmax at position 1 (~0.27),
  inconsistent with C&Z's pure-recency-init prediction.
- C&Z's hierarchical retrieval (Eqs 12–15) already has the conceptual
  ingredients: list-level activations select a list, then item-level
  recall proceeds from the list's beginning-of-list context. C&Z just
  applies this AS A FALLBACK after item-level retrieval fails. The fix
  is to apply it AT INITIATION too.

**Mechanisms added** (`_likelihood_core_mstcm.py`):
1. **Per-storyline contexts** (one per category, indexed via the
   `category` column). Each storyline drifts ONLY when items from that
   storyline are encoded; inactive storylines are frozen. (v1 §3.2.)
2. **λ storyline-return reinstatement** at encoding (v6 Eq 6). When a
   storyline resumes after interruption, blend cached + current
   storyline contexts.
3. **τ storyline-initiation mixture** at recall onset (NEW). With
   probability τ, sample storyline ŝ ∝ softmax(k · M^SC · c^list_end);
   reactivate item context to c^story_ŝ; sample first recall under
   standard C&Z dynamics. With probability 1−τ, standard C&Z recency
   initiation. After first recall, both routes use C&Z dynamics
   (β_rec drift + ε_d stopping + e_start fallback).

**Implementation**: shared math factored into `_likelihood_shared.py`;
MS-TCM reuses C&Z's `_compute_ll_per_T` for the after-first-recall
sequence likelihood (only the initial c_ret seed differs between
routes). Reduces exactly to C&Z when K=1, τ=0, λ=0.

**FRFR-category fit**: LL = −10502.16 (ΔLL=+197.66 over C&Z;
ΔAIC=−391.32). MLE finds τ = 0.408 (substantial mixture weight).
Notably β_list collapses to ~0 — with τ active, the list-level context
becomes redundant (parameter substitution).

**Empirical evaluation** (figs/source/fig_analyses):
- ✓ pFR: now produces a primacy spike at sp=1 (~0.10 model vs ~0.27
  observed). C&Z baseline produced 0 there.
- ✓ lag-CRP: unchanged from C&Z (good — both within observed CI).
- ≈ SPC: similar shape to C&Z, both within observed CI.

**Remaining gap (motivates Iteration 2)**: the SPC has a "scalloped"
shape — local maxima at sp = 1, 5, 9, 13, local minima around
3–4, 7–8, 11–12. Period = 4, which matches FRFR-category's 4 items
per category. Each category-onset boundary acts as a "mini list-onset"
with enhanced recall.

Neither C&Z's bowl nor MS-TCM's slightly-modified bowl reproduces the
scallop pattern. The current MS-TCM uses storyline contexts at recall
INITIATION but does not produce within-category primacy at every
storyline boundary because the encoding-time c_item drift is still
fully continuous across category boundaries (no within-category reset).

---

## Iteration 2 (proposed) — Per-storyline hierarchical contexts plus a global cross-storyline context

**Empirical motivation**:
- The 4-period scalloped SPC visible in observed FRFR-category data:
  positions 1, 5, 9, 13 are local recall maxima; positions 3-4, 7-8,
  11-12 are local minima.
- This is a *within-storyline* primacy effect: each category onset
  acts like a list-onset, producing enhanced recall of the first item
  of each category block (and a smaller recency bump for the last
  item).
- Current MS-TCM has per-storyline contexts but they're used only at
  retrieval initiation. There's no encoding-time mechanism that
  produces "fresh start" boundary effects within a category sequence.

**Conceptual core** (Jeremy's framing):
> One set of hierarchical context representations (as in C&Z) for each
> storyline, plus an overarching set of hierarchical representations
> for time in general (storyline agnostic). The overarching
> representation allows storylines to be related to each other through
> associations with that overarching context.

**Translation to a candidate architecture**:

For each list with K storylines, maintain TWO nested hierarchies:

(1) **Per-storyline hierarchy** (one copy of C&Z's two-level model
    per storyline):
    - c^item_s: storyline-s's item-level context (drifts only when
      storyline-s is being encoded; frozen otherwise)
    - c^list_s: storyline-s's list-level context (slower drift, only
      when storyline-s is active; frozen otherwise)
    - Each storyline has its own M^FC_pre/exp, M^CF_exp, M^lists
      pair, scoped to that storyline's items.
    - At a storyline boundary (entering storyline s), the standard
      C&Z hierarchical retrieval reset rule applies: c^item_s is
      reactivated to c^list_s's beginning-of-list context.

(2) **Global hierarchy** (one storyline-agnostic copy of C&Z that
    spans the full list):
    - c^item_G: drifts at every encoding step regardless of
      storyline (continuous time across the whole list)
    - c^list_G: drifts at every step, slower rate
    - Global associative matrices: M^FC_exp_G, M^CF_exp_G,
      M^lists_G (one entry per storyline, holding c^list_s_end as
      that storyline's list-level context; the global hierarchy
      treats storylines as its "lists").

(3) **Composite encoding context** (item gets encoded with the union):
    For an item i from storyline s at global step t:
    - c_compIN_i = w_S · c^item_s_i + w_G · c^item_G_t
      (drift target uses both)
    - The associative matrix updates use this composite, so each
      item is associated with both its storyline's local context
      AND the global temporal context.

(4) **Retrieval routing** (the τ-mixture extends to a 3-route choice):
    - Route 1 (recency, prob 1−τ_S−τ_G): standard end-of-list cue
      using c^item_G_W.
    - Route 2 (storyline-init, prob τ_S): sample ŝ from the global
      hierarchy's M^lists_G (storylines are entries here);
      reactivate to that storyline's beginning-of-storyline
      context c^list_ŝ_0; recall items via that storyline's
      M^CF_exp_s.
    - Route 3 (within-storyline-onset-init, prob τ_G·...): could
      be subsumed by Route 2 — when ŝ is selected, the
      reinstatement to c^list_ŝ_0 produces within-storyline
      primacy directly.

**Mechanism producing the scalloped SPC**:
- Each storyline's per-storyline hierarchy contributes its own
  primacy effect via its own list-level fallback (every time a
  storyline-s item is recalled, its retrieval can fall back to
  storyline-s's c^list_s_0, which favors the FIRST item of
  storyline s).
- Across the full list, storylines start at positions 1, 5, 9, 13
  (in blocked early lists), so the per-storyline primacy effects
  produce local SPC maxima at those positions.
- Within a storyline, the standard recency-then-primacy fallback
  pattern replays.

**New parameters** (vs Iteration 1):
- w_S, w_G: composite-context mixing weights (constraint:
  w_S + w_G = 1, so 1 free parameter).
- That's the minimum delta. We can also revisit:
- β_enc^S vs β_enc^G: per-storyline vs global encoding drift rates
  (could be different — global drift might be slower since global
  context spans more events).
- λ already exists for storyline-return reinstatement.

**Reductions** (sanity):
- w_S = 0: model reduces to current Iteration-1 MS-TCM (only global
  context drives retrieval; per-storyline contexts are vestigial).
- w_G = 0: model reduces to a "fully partitioned" model where each
  storyline is encoded/recalled independently — no cross-storyline
  associations, no scallop integration.
- K = 1: per-storyline = global; w mixture is irrelevant. Reduces
  to C&Z.

**What this predicts (and what we should look for in fits)**:
- SPC scallops at category boundaries (the main target).
- pFR continues to show primacy spike at sp=1 from storyline-init.
  Magnitude could be modulated by w_S vs w_G (high w_S → stronger
  storyline-init bias → bigger pFR spike).
- Lag-CRP: should retain forward-asymmetric peak at +1 within
  storylines; potentially weaker between-storyline transitions.
  Could examine same-category vs different-category transitions
  separately as a diagnostic.

**Implementation plan** (when ready):
1. Add `mstcm_v2` core with per-storyline + global hierarchies
   (likely `_likelihood_core_mstcm_v2.py`). Heavy reuse from
   v1 + shared.
2. Composite encoding: per-step, drift c^item_s only on
   storyline-s items; drift c^item_G on every step.
3. Composite retrieval: each route uses a different (c_ret_init,
   M^CF_exp) pair. Retrieval marginalizes over routes via softmax.
4. Add w_S/w_G mixing weight as a fitter parameter.
5. Oracle tests + backend parity (mirror v1 pattern).
6. Fit + figure overlay vs both C&Z and v1 MS-TCM.

**Risks / things to watch**:
- Identifiability: w_S, w_G, τ, β_list, β_list_s, λ are all
  related. Need diagnostics that each parameter is doing
  distinct work.
- Computational cost: K storylines × (item, list) = 2K context
  vectors instead of 2. For FRFR-category K=4 → 8 context
  vectors per list. Manageable.
- The "scallop" SPC could potentially also be explained by
  an item-level β_enc that resets at storyline boundaries
  (a simpler mechanism). Worth checking that the
  per-storyline-hierarchy story isn't overkill compared to
  a "boundary-reset" mechanism.

---

## Iteration N (placeholder)

When we propose further changes, append a new `## Iteration N`
section here. Each section should follow the same template:
- Empirical motivation
- Mechanisms added (or removed)
- Reductions / sanity properties
- What this predicts
- Implementation plan
- Risks
