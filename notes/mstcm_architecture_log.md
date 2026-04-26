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

## Iteration 2 (refined) — Strict hierarchical recall: global cues storyline, storyline cues events

**Empirical motivation**: same as Iteration 2 (proposed) above —
SPC scallops at category boundaries (peaks at 1, 5, 9, 13).

**Note on FRFR-category structure**:
- Lists 0–7 (early): items grouped by category but with **variable
  category order per list** (not strict ABCD blocks; categories
  alternate at the 1–2 item granularity within early lists).
- Lists 8–15 (late): random order (NOT systematically interleaved).

The observed average-across-lists scallop pattern reflects that on
average, category-boundary positions cluster near 1, 5, 9, 13 across
lists, but individual lists have varying boundary structure.

### Architecture (refined per Jeremy's specification)

**Two-level strict hierarchy**, where storylines act as "items"
for the global level and events are sub-items within each storyline:

```
GLOBAL level                  STORYLINE-s level
c^global, c^list_G            c^story_s, c^story_list_s    (one per storyline)
M^FC_G, M^CF_G                M^FC_s, M^CF_s
M^lists_G                     -

drifts on EVERY item          drifts ONLY on storyline-s items
```

**Encoding (per Jeremy: option Q1=b, Q2=iii)**:
- Every item's pre-experimental input drives BOTH branches:
  - Global branch: c^global drifts (via β_enc^G), c^list_G drifts
    (via β_list^G), update M^FC_G and M^CF_G with this item.
  - Storyline branch: only the active storyline's c^story_s drifts
    (via β_enc), only its c^story_list_s drifts (via β_list); other
    storylines stay frozen. Update M^FC_s and M^CF_s with this item.
- λ storyline-return reinstatement still applies at the storyline
  level (when a storyline resumes after a gap, blend cached + current
  via λ).
- M^lists_G is a (K, d_global) matrix. At storyline departures or
  list end, write c^story_list_s_end into M^lists_G[s] (so the global
  level can retrieve a storyline's cached list-context).

**Retrieval (refined per Q3)**: ONE retrieval procedure with a τ
mixture between two routes:

**Route α — recency (prob 1 − τ)**:
- Cue: c^global_W (end-of-encoding global context).
- Score against M^CF_G — produces a per-item activation distribution
  via softmax(k · M^CF_G · c_ret_G).
- Sample first recall, drift c^global_ret, repeat. Stopping rule per
  C&Z. On stop: terminate.
- This is a flat, global-only retrieval — NO storyline lookup at all.
  Pure recency-driven, like base C&Z without hierarchical fallback.

**Route β — strict hierarchical (prob τ)**:
- Step 1 (global cues storyline). a^story = M^lists_G · c^global_W;
  sample storyline ŝ ∝ softmax(k · a^story).
- Step 2 (set storyline-level cue per Jeremy's preference, option
  ii from "what does c^story_ŝ look like at retrieval"):
  c_ret_s = e_start (the within-storyline beginning-of-context). This
  produces within-storyline PRIMACY (favors the first storyline-ŝ
  item), matching the observed scallop peaks at 1, 5, 9, 13.
- Step 3 (within-storyline retrieval). Score against M^CF_s with
  c_ret_s; sample, drift, stopping rule. Already-recalled mask per
  C&Z (across whole list, not just storyline).
- Step 4 (on storyline-ŝ stop, decide what to do):
  **Default: option B** — re-run Step 1 with c^global_ret (drifted
  by β_rec during the storyline-ŝ recalls), but with storyline ŝ
  REMOVED from the candidate set ("exhausted" — once a storyline
  retrieval has stalled, don't reselect it). If no candidate
  storylines remain, terminate.
- The pattern of recall under route β is: cluster within ŝ_1,
  switch to ŝ_2, cluster within ŝ_2, etc. This produces strong
  same-category lag-CRP transitions and clear category boundaries.

After the first recall in EITHER route, β_rec drift updates c^global
(and any active c^story) toward the recalled item's c^IN_rec.

**Open questions where Jeremy's input matters most**:

[OQ1] Should route α use the GLOBAL associative matrix (M^CF_G) or
ALSO consult per-storyline matrices? Currently I'm specifying
M^CF_G only, so route α is a pure-global retrieval with no
storyline involvement. Alternative: route α uses a composite
M^CF that's the union of M^CF_G + Σ_s M^CF_s. The first is
cleaner for identifiability; the second has more total
associations available at retrieval but blurs the route
distinction.

[OQ2] In route β step 4 (storyline exhaustion), should we ALSO
update M^lists_G to reflect that the storyline was exhausted
(e.g., zero out that row)? Or just track it via a separate
"exhausted" mask? Mask is cleaner.

[OQ3] In route β step 2, the "c_ret_s = e_start" choice produces
pure within-storyline primacy. An alternative is c_ret_s =
M^lists_G[ŝ] (the cached storyline-list context at storyline
ŝ's last departure). That's the v6 Eq 6 / C&Z Eq 14 style
reinstatement. Should give a more "C&Z-like" within-storyline
recency-then-primacy curve for that storyline. We picked
e_start for now to maximize the scallop signal.

**Parameter inventory** (vs Iteration 1):
- ADD β_enc^G: separate global-level encoding drift rate. 10 params total.
- DEFER per-storyline β_enc_s: too many degrees of freedom for now;
  reintroduce when fitting heterogeneous-storyline datasets
  (short-story vs novel vs film comparison).

**Reductions** (sanity):
- K = 1: storyline level = global level; M^lists_G has one row.
  Should reduce to C&Z 2025 (with τ acting as the original v1 τ).
- τ = 0: Iteration 2 reduces to "C&Z without hierarchical fallback"
  — pure global-context retrieval.
- λ = 0: storyline contexts don't reinstate at returns (only matters
  in interleaved encoding, which late FRFR lists lack systematically).

**What this predicts (and what we should look for in fits)**:
- ✓ SPC scallops at storyline boundaries from route β within-storyline
  primacy.
- ✓ pFR primacy spike at sp=1 from route β when storyline ŝ_1 happens
  to be the first-encoded storyline.
- ✓ Same-category lag-CRP > different-category lag-CRP, more
  pronounced than under iteration 1 (route β chains within a
  storyline before switching).
- New diagnostic: FRP (probability of first-recall by category) —
  if route β fires (prob τ) and selects ŝ uniformly, then ~τ/K mass
  goes to each category's first item. Compare across categories.

**Risks / things to watch**:
- (i) Route β is recursive (after a storyline stops, we go back to
  global). Need to bound the number of route-β iterations to keep
  computation tractable. Could cap at K (one round-trip per storyline)
  since once exhausted, storylines drop out.
- (ii) The "c_ret_s = e_start" choice (OQ3) is the simplest but might
  over-predict within-storyline primacy at the expense of recency
  within a storyline. Could fit and check residuals.
- (iii) Per-storyline associative matrices add 2K matrices of size
  (d, W_s) each, where W_s is items in storyline s. For FRFR-category
  K=4 with 4 items each, that's 8 small matrices. Cheap.
- (iv) The likelihood now needs to marginalize over (route, ŝ) for
  the first recall AND over which storyline each subsequent recall
  was "produced from" (since under route β, intermediate recalls
  could come from any storyline that has been visited and not
  exhausted). This is more complex than the current latent T.

**Implementation plan (when we agree on OQ1, OQ2, OQ3)**:

The likelihood-marginalization complexity in Risk (iv) is the main
concern. A practical simplification: assume the route is fixed for
the entire recall sequence (chosen at recall onset), so route α
generates ALL recalls or route β generates ALL recalls. Then the
likelihood is τ · LL_β + (1-τ) · LL_α at the top level. Under route
β, the latent is the sequence of storyline visits ŝ_1, ŝ_2, ...,
which can be inferred from the recalls (each recall's storyline
identity is observed via cat_indices). So route β's LL conditional
on the storyline-visit sequence is straightforward; the marginal
just sums over storyline orderings consistent with the observed
recall sequence — and there's typically only ONE such ordering
(the order in which storylines first appear in the recalls). So the
marginalization is small.

This dramatically simplifies the implementation. I'll write up the
detailed pseudo-code once OQ1-OQ3 are resolved.

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
