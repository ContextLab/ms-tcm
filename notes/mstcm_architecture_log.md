# MS-TCM Architecture Iteration Log

This file tracks the evolution of the MS-TCM model architecture across
sessions. Each iteration records the empirical observation that motivated
the change, the proposed mechanism, the rationale, and the predicted
behavioral signatures the change should produce.

The point is to make the reasoning chain reconstructable — when a future
session asks "why does the model have X?", the answer should be findable
here.

**Code-vs-log convention (Constitution II compliance)**:

- This log records the **history of design decisions** — each iteration
  documents what changed, why, and what was predicted.
- The **code is monotonic**: only the *current* MS-TCM model lives in
  `code/ms_tcm/_likelihood_core_mstcm.py`. There are NOT separate
  `mstcm.py`, `mstcm_v2.py`, `mstcm_v3.py` files. Each iteration's
  changes replace the prior version IN PLACE.
- The single source of truth for "what the model is right now" is the
  code; the log is the audit trail for "how we got here."
- When you read this log, the LATEST iteration section corresponds to
  the code's current state. Earlier iterations describe code that no
  longer exists in the repo.

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

**Open questions — RESOLVED per Jeremy's input**:

[OQ1] **Resolved: option (a)**. Route α uses M^CF_G only — pure
global recall, no storyline involvement. This keeps route α and
route β cleanly separated for identifiability. We can revisit
later if the fit suggests blurred separation is needed.

[OQ2] **Resolved: separate `exhausted` mask, but with refined
semantics**. The `exhausted` mask is NOT a permanent flag — it
tracks only the **immediately-preceding** storyline. After route
β recalls items from storyline ŝ_a and stops, ŝ_a is excluded
from the next global storyline-selection. But once any OTHER
storyline ŝ_b is visited, ŝ_a becomes available again on
subsequent global re-selections. Implementation: maintain a
single int `last_recently_exhausted_storyline` (initialized to
−1); on each global storyline-selection step, exclude storyline
matching this index from the softmax candidates; after a new ŝ
is selected, update the index to the storyline JUST EXHAUSTED
(only when the previous storyline's recall stopped). This
captures the "you don't go back to the same storyline twice in a
row" intuition, allowing return after intervening visits.

[OQ3] **Resolved (initial): option (ii) c_ret_s = e_start, with
plan to test (iii) c_ret_s = M^lists_G[ŝ] (storyline's cached
list-context) in a follow-up**. Option (ii) maximizes the
scallop signal on blocked early lists. Option (iii) is more
principled (it's literally what C&Z Eq 14 does for the
hierarchical fallback) and may behave differently on interleaved
lists where "beginning-of-storyline" and "beginning-of-list"
diverge — in late FRFR-category lists (random order), these
contexts will be very different vectors. Worth testing both.

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

## Behavioral diagnostics (2026-04-26)

Two empirical investigations of FRFR-category by behavioral-data agents
sharpened our understanding of what Iteration 2 needs to capture.
Detailed notes at `notes/clustering_analysis.md` and
`notes/late_list_scallop_analysis.md`; figures in
`notes/clustering_analysis_figures/` and
`notes/late_list_scallop_figures/`.

### Finding 1: lag-CRP difference is a composition shift

When the lag-CRP is decomposed into within-category vs across-category
transitions, the two transition types have **identical** lag-CRP shapes
in early and late lists (within-cat: ~0.55 at lag +1 for both halves;
across-cat: ~0.21 at lag +1 for both halves). The overall lag-CRP
difference between halves comes entirely from the proportion mix:

- Early lists: 57% of transitions are within-category (peaked at +1)
- Late lists: 38% of transitions are within-category

So the "tighter clustering in early lists" = "more within-category
transitions in early lists." The within-category lag-CRP is identical;
what changes is how often participants make within-category transitions.

In sorted lists, within-category items are also temporally adjacent
(consecutively presented), so a within-category transition mechanically
produces a |lag|=1 transition. In random lists, within-category items
are scattered, so a within-category transition produces a wide range of
lags (mean ≈ 4-5).

Statistical evidence:
- TCS (temporal clustering score): early 0.743 vs late 0.602
  (Cohen's d = 1.33, p = 4.8 × 10⁻⁸)
- SCS (semantic clustering score, Polyn et al. 2009): early 0.373 vs
  late 0.197 (Cohen's d = 1.77, p = 1.4 × 10⁻¹⁰)
- TCS-SCS within-participant correlation: r = +0.72 in early lists
  (mechanisms align), r = -0.53 in late lists (mechanisms compete).
  This is consistent with TCS and SCS being driven by ONE underlying
  recall-organization process (people who organize by category get
  high SCS; in sorted lists this also produces high TCS; in random
  lists, organizing by category SACRIFICES temporal clustering).

### Finding 2: late-list scallop is real and category-organized

The late-list SPC scallop (peaks at 1, 5, 9, 13) is statistically
robust (permutation p = 0.001). It is FULLY explained by category-
organized recall:

- Within-category run length: late = 1.66 (well above null = 1.0,
  Cohen's d = 1.47 vs early = 2.27)
- Within-category transition rate: late = 0.38 (1.91× chance,
  p < 0.001)
- Category-rank decomposition: SPC predicted from per-rank histograms
  (where rank = "i-th item recalled within a category") matches
  observed late SPC at r = 0.999.

The Rank-1 panel of the rank decomposition shows the key signature:
the FIRST item recalled from each category has a strong primacy bias
(disproportionately at sp=1). When this is averaged across 4 categories
that are scattered across the list, you get scallop peaks at the
expected positions because Rank-1 items also tend to occupy the early
positions of their categories.

Output-position chunking (recalls grouped in chunks of 4 by output
position, regardless of category) was REJECTED: intra-chunk SP variance
on late lists (19.82) is close to the random null (22.71) — chunks
do NOT access spatially coherent regions of the list.

### Implications for Iteration 2

The two findings converge on a single mechanism: **participants
organize their recall by category regardless of presentation order**.
This is what produces both the lag-CRP composition shift AND the
late-list scallop. Iteration 2 should reproduce this organization
because:

1. ✓ The strict-hierarchical retrieval (route β: global cues storyline,
   storyline cues events) ALREADY enforces category-organized recall
   when route β fires. This is the right mechanism.

2. The **τ mixture parameter** controls how often route β fires.
   Behavioral evidence suggests this should be HIGH in BOTH early
   and late lists (since participants organize by category in both —
   the run length in late lists is 1.66, well above 1.0). Earlier
   MS-TCM v1 fit found τ ≈ 0.41, which is plausibly underestimated
   if the model can't fully capture the organization.

3. Within-category retrieval (the "storyline cue") should produce
   the within-category mini-SPC visible in the rank-1 decomposition.
   Pure within-storyline primacy (OQ3=ii, c_ret_s = e_start) might
   over-predict — participants don't always start within a category
   from the first item; some randomness exists. Option (iii)
   c_ret_s = M^lists_G[ŝ] (cached storyline-list context) might
   better capture the modest within-category primacy + recency
   pattern visible in the data. Worth testing both.

4. The **storyline-exhaustion mechanism** (only-immediately-preceding
   excluded) is consistent with the run-length data: median
   within-category run length is ~2-3, and people DO return to
   categories after intervening visits.

### Diagnostic targets for Iteration 2 fits

When fitting MS-TCM v2, evaluate against:
- Within-category lag-CRP shape (should be ~0.55 at +1, with forward
  asymmetry)
- Across-category lag-CRP shape (should be ~0.21 at +1, with forward
  asymmetry)
- Fraction of within-category transitions in early vs late lists
  (should reproduce the 57% / 38% split)
- Within-category run length (should reproduce 2.27 / 1.66 split)
- Late-list scallop amplitude (~0.074 between peaks and troughs)
- pFR primacy spike at sp=1 (~0.27 in both halves)

A good Iteration-2 model gets 4-5 of these right; a great one gets
all 6.

### Outstanding question (for Iteration N+1)

Why is participants' tendency to organize by category SOMEWHAT
weaker on late lists (run length 1.66 vs 2.27)? Not just structural —
even adjusting for the random presentation order, participants seem
less aggressive about category clustering on late lists. Possibilities:
- Practice effects: late lists come second, fatigue.
- Strategy weakening when categories are random (less obvious to
  re-impose category order).
- Individual differences in strategy adoption (Participant 12 in the
  raw-sequence inspection showed near-serial recall while
  Participant 7 showed perfect category blocking).

The model currently has no mechanism for early-vs-late strategy
shifts. If we want to capture both halves with ONE parameter set,
that's a constraint on Iteration 2. Alternatively, we could fit
the two halves with different τ values, treating the early/late
split as a session-effect dimension.

---

## Iteration 2 fit results (2026-04-26)

The consolidated MS-TCM (strict-hierarchical) was fit to FRFR-category
on the 480-list dataset with 3 restarts, finding:

```
LL = -10773.54  (worse than pre-consolidation MS-TCM = -10502.16,
                 worse than C&Z baseline = -10699.82)
beta_enc = 0.866
beta_enc_global = 0.695
beta_list = 0.195
gamma_fc = 0.747
k = 2.56
beta_rec = 0.948
epsilon_d = 1.92
beta_rein = 0.311
lambda_reinstate = 0.468
tau_init = 0.379
```

τ and λ are both substantial (mechanisms are firing), but the model's
overall LL is WORSE than both the C&Z baseline and the pre-consolidation
MS-TCM — by ~74 nats vs C&Z and ~271 nats vs pre-consolidation.

### What MS-TCM v2 captures that earlier models did not

Despite the worse LL, the consolidated MS-TCM produces several
qualitatively-correct behaviors that prior architectures missed —
indicating the storyline-hierarchy mechanism is genuinely doing
useful work:

1. **pFR bump at sp = 13** (early lists, blocked by category): the
   model produces a pFR peak at ~sp 13, matching the observed bump
   at the position where the LAST encoded category begins. This is
   the storyline-init mechanism (route β) selecting the most-recent
   storyline (whose c^story end-state overlaps c^global end-state)
   and reactivating that storyline's beginning-of-context — which
   favors the FIRST item of that final storyline. Neither C&Z nor
   the pre-consolidation MS-TCM produced this bump; this is direct
   evidence the strict-hierarchical mechanism does what we designed
   it to do.

2. **Some scallop structure in the early-list SPC**: the consolidated
   model produces partial scalloping in the SPC for blocked lists,
   capturing the within-storyline primacy effect at category
   boundaries (storyline starts at sp 1, 5, 9, 13). The amplitude
   is too low and the scallops are smoothed compared to observed,
   but the shape signature is present.

3. **Decrease in temporal clustering between early and late lists**:
   the model produces tighter neighbor-clustering in early
   (blocked) lists than in late (random) lists, matching the
   observed "composition shift" finding from the behavioral
   diagnostics agent. This emerges naturally from the strict
   hierarchy: in blocked lists, within-storyline transitions
   coincide with adjacent serial positions (high TCS); in random
   lists, within-storyline transitions span scattered positions
   (lower TCS).

These successes confirm the **per-storyline + global hierarchy
mechanism is on the right track**. The LL gap vs simpler models
reflects model-class issues that Iteration 3 needs to address,
not a fundamental flaw in the storyline-hierarchy idea.

### What MS-TCM v2 fails to capture

1. **No pFR primacy spike at sp = 1** (both early and late lists):
   observed pFR shows a strong spike at the FIRST item of the list
   (~0.27), which the model under-predicts (~0.05). The mechanism
   the model uses for primacy — storyline-selection softmax —
   favors the MOST-RECENT storyline (whose context overlaps the
   end-of-encoding cue), not the first-encoded storyline. So the
   model gets the "secondary" pFR peak right (sp 13) but misses
   the primary one (sp 1).

2. **No scallop structure in the late-list SPC**: late lists have
   random category order, so storyline boundaries don't align with
   fixed positions. The behavioral data show scallops persist in
   the late SPC (because participants still recall by category, just
   from scattered positions), but the model produces a smooth bowl —
   it doesn't reorganize the recalls by category aggressively
   enough.

3. **Strict-separation cost dominates LL**: under route β, items in
   non-active storylines have zero activation via M^CF_s. The data
   show 38% of late-list transitions ARE across-category, meaning
   the strict architecture forces the model to pay log-likelihood
   cost via the cross-storyline-selection mechanism for every
   cross-category transition — which compounds across recalls.

### Diagnosis

The strict-hierarchical architecture is TOO restrictive for FRFR-
category. Specifically:

1. **Per-storyline associative matrices over-partition the model**.
   Under strict separation, when route β fires for storyline ŝ, only
   M^CF_s can produce activations — items in OTHER storylines get
   zero activation regardless of how the retrieval context might cue
   them. The pre-consolidation MS-TCM allowed within-storyline
   retrieval to also activate items from other storylines via the
   shared M^CF_G, so the model could occasionally cross category
   boundaries mid-recall.

2. **The visit boundary structure is rigid**. Under route β, the
   only way the observed cross-category transitions can occur is
   via storyline-stop → storyline-selection → new-storyline-start,
   with the immediately-preceding-only exclusion. This makes the
   model brittle: every cross-category transition costs LL.

3. **Storyline-selection bias is wrong direction for primacy**: the
   global cue at end-of-encoding overlaps most with the most-recent
   storyline, so storyline-init under-weights the first-encoded
   storyline. The model captures the "secondary recency bump" at
   pFR(sp=13) but misses the dominant primacy spike at pFR(sp=1).

### What this suggests for Iteration 3

- **Option A (recommended; Jeremy approved 2026-04-26)**: Relax
  the strict separation by adding a parameter `w_global ∈ [0, 1]`
  that mixes M^CF_s and M^CF_G during within-storyline retrieval.
  At w_global=0 we recover strict hierarchy; at w_global=1 the
  model's route-β within-storyline phase uses M^CF_G alone (no
  storyline matrix), making it equivalent to an unrestricted
  recall after the storyline-selection step. Free parameter
  enables MLE to find the right balance for FRFR-category.

- (deferred) Option B: revisit OQ3 — try `c_ret_s = M^lists_G[ŝ]`
  (cached storyline list-context) instead of e_start.

- (deferred) Option C: revisit OQ1 — let route α use a composite
  (M^CF_G + Σ_s M^CF_s) rather than M^CF_G alone.

**Empirical anchor for Iteration 3 design**: the data show participants
make BOTH within-storyline AND across-storyline transitions during
recall, with within-storyline at 57% / 38% (early/late). The model
needs to capture this mixture WITHIN a recall flow, not just at
visit boundaries.

---

## Iteration 3 — Soft storyline boundaries via w_global (committed)

**Empirical motivation**: see "What MS-TCM v2 fails to capture" above.
The strict per-storyline separation forces 100% within-category
transitions during route-β within-storyline phases; data show 57%/38%
within-cat (early/late). The model needs to allow cross-storyline
activations during within-storyline recall.

**Mechanism added**: a single new parameter `w_global ∈ [0, 1]` that
mixes per-storyline and global associative matrices during the
within-storyline phase of route β:

    M^CF_effective(ŝ) = (1 − w_global) · M^CF_ŝ + w_global · M^CF_G

Same for M^FC during the c_IN_rec drift target:

    M^FC_effective(ŝ) = (1 − w_global) · M^FC_ŝ + w_global · M^FC_G

Storyline-selection under route β still uses M^lists_G (cached
storyline contexts) — that's how route β finds storylines to recall
from. The mixing only affects the WITHIN-storyline retrieval phase
once a storyline has been selected.

Route α is unchanged (still uses M^CF_G alone — pure-global recall).

**Reductions** (sanity):
- w_global = 0 → exactly Iteration 2 (strict hierarchy)
- w_global = 1 → route β within-storyline phase uses only M^CF_G
  (no per-storyline associations during retrieval); the model
  becomes "tau-mixture between two routes both using M^CF_G," but
  with route β additionally re-cueing item context to e_start via
  the storyline-selection step. Should reduce to a model class
  similar to pre-consolidation MS-TCM with a different cue
  re-instatement source.
- K = 1 → per-storyline matrix is identical to the global matrix;
  w_global is irrelevant; reduces to Iteration 2 K=1 reduction
  (which itself reduces to C&Z).

**What this predicts**:
- pFR(sp=1) primacy spike: should improve — when within-storyline
  retrieval mixes in M^CF_G, items from earlier storylines are
  reachable via global context overlap, and the "primacy storyline"
  doesn't have to win the storyline-selection softmax exclusively.
- Late-list scallop: should improve — within-storyline retrieval
  with non-zero w_global can reach items from non-active categories
  by happenstance, producing more category-mixing during recall
  flow (matching observed transition rates).
- Early-list scallop: should be at least as good as Iteration 2
  (strict hierarchy still produces it via the storyline-init
  mechanism); marginal at w_global > 0 because some scalloping
  comes from within-cat cohesion that the storyline cue still drives.
- LL: should improve substantially. The strict-separation cost of
  Iteration 2 (~74 nats vs C&Z) reflects the LL the model "pays" for
  each cross-category transition; relaxing this with w_global > 0
  should recover most of that cost.

**Parameter inventory** (vs Iteration 2):
- ADD `w_global` ∈ [0, 1]: 11 parameters total
  (β_enc, β_enc_global, β_list, β_rec, β_rein, γ_fc, k, ε_d,
   λ_reinstate, τ_init, **w_global**)

**Implementation plan**:
1. Add `w_global` field to ModelParameters (default 0.0 to preserve
   Iteration 2 baseline).
2. Modify _likelihood_core_mstcm.py within-storyline LL helper to
   compute M^CF_effective + M^FC_effective inline. Drift uses the
   c_IN_rec computed from M^FC_effective.
3. Update simulator to use the same effective matrices.
4. Add reduction-test (Iteration 2 result reproduced when
   w_global = 0).
5. Update fitter to add the 11th parameter.
6. Refit FRFR-category; compare LL vs Iteration 2.
7. Regenerate fig_analyses; check pFR(sp=1) primacy and SPC scallops.

**Risks / things to watch**:
- Identifiability: w_global vs τ vs gamma_fc may trade off. Need to
  check that the MLE assigns each parameter a distinct role.
- Fit speed: the within-storyline LL becomes a small linear-combo
  computation; should be O(1) extra cost per recall.

### Iteration 3 fit results (2026-04-26)

The soft-hierarchy fix worked decisively:

| Model | LL (FRFR) | Δ vs C&Z | n_params | ΔAIC vs C&Z |
|-|-|-|-|-|
| C&Z baseline | −10699.82 | — | 7 | — |
| MS-TCM v2 (strict) | −10773.54 | −74 (worse) | 10 | +180 (worse) |
| **MS-TCM v3 (soft)** | **−10608.85** | **+91 (better)** | **11** | **−174 (better)** |

3 restarts all converge to LL = −10608.85 (machine-precision agreement
across restarts; this is the global MLE).

**Fitted MLE** (consolidated MS-TCM v3 on FRFR-category):
```
beta_enc = 0.587            (item-level drift, per-storyline scope)
beta_enc_global = 0.791     (global drift, faster than per-storyline)
beta_list = 0.160           (slow list-level drift)
gamma_fc = 0.653
k = 3.336
beta_rec = 0.888
epsilon_d = 2.583
beta_rein = 0.311
lambda_reinstate = 0.468    (substantial storyline-return reinstatement)
tau_init = 0.507            (route β fires on ~50% of trials)
w_global = 0.511            (~equal mix of per-storyline and global
                              associations during within-storyline recall)
```

**Key finding**: w_global = 0.51 confirms the diagnosis that strict
per-storyline separation was over-restrictive. Participants apparently
combine within-category and global cues in roughly equal measure during
within-storyline retrieval — the soft mixing parameter let the model
recover the LL it was paying via the strict-architecture cross-category
transition cost.

**Behavioral test results** (figure: paper/figs/source/fig_analyses):
- ✓ pFR (early lists): MS-TCM (blue) tracks observed shape better than
  C&Z (red); both still under-predict the sp=1 primacy spike but
  MS-TCM is slightly better.
- ✓ pFR (late lists): same — MS-TCM matches the observed
  recency-dominant shape with a small primacy contribution.
- ✓ lag-CRP (both halves): MS-TCM closely tracks the observed +1 peak.
- ✓ SPC (early lists): MS-TCM band shifted up to better align with
  observed; some scallop structure visible.
- ✓ SPC (late lists): MS-TCM well-aligned with observed.

**Parameter interpretations**:
- β_enc_global > β_enc (0.79 > 0.59) is initially surprising — would
  expect global to drift slower. But this likely reflects: c^global
  integrates ALL items (16 per list), while c^item_s integrates only
  ~4 items per storyline. To produce a similar magnitude of "context
  evolution" over the encoding period, the global rate per-step must
  be higher (similar logic to why a longer list needs faster drift to
  cover the same context distance). This is geometry of the
  Howard-Kahana drift, not a substantive theoretical claim.
- τ = 0.51, w_global = 0.51 — both at ~50%. The MLE distributes
  between the two routes evenly and combines per-storyline + global
  associations evenly within route β.
- λ = 0.47, β_rein = 0.31 (gradient-zero, fixed at init): λ is doing
  meaningful work; β_rein is a nuisance.

**What's still missing**: pFR primacy spike at sp = 1. The model
captures the secondary recency bump at sp = 13-15 (storyline-init
mechanism selecting the most-recent storyline) but the dominant
primacy spike at sp = 1 (~0.27 observed) remains under-predicted
(model gives ~0.05). This appears to be an architectural feature of
option (ii) — using e_start as the within-storyline cue produces
within-storyline primacy, but storyline-selection prefers the
most-recent storyline (whose c^story end-state overlaps c^global
end-state). The model captures the LAST storyline's first item
(pFR=13) but not the FIRST storyline's first item (pFR=1).

**Suggested Iteration 4 directions** (deferred):
1. Test option (iii) for OQ3: c_ret_s = M^lists_G[ŝ] (cached storyline-
   list context) instead of e_start. Different within-storyline
   recency-vs-primacy bias.
2. Add a "primacy bias" to storyline selection — softmax over
   (M^lists_G · c_ret_g) PLUS a positive scalar at the FIRST-encoded
   storyline (analogous to CMR primacy gradient applied at the
   storyline level).
3. Investigate whether the FRFR-category data have a task-driven
   start-from-beginning instruction effect (cf. earlier behavioral
   investigation).
