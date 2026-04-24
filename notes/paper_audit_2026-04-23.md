# Paper audit — 2026-04-23

This note catalogues every issue found during the paper-update audit. Items are grouped by severity. The audit was done by verifying paper claims against (a) the actual FRFR-category data, (b) the actual model outputs, (c) Cornell & Zhang 2025 (`notes/CornZhan25.pdf`), and (d) visual review of every figure.

## Severity 1 — scientific correctness

### Fit produces worse behavioral curves than defaults

**Finding.** The MS-TCM MLE fit (10-participant subset, 3 restarts) converges to parameters (β_enc=0.63, β_story=0.63, γ_fc=0.37, k=3.83, λ=0.02, β_rec=0.67, ε_d=2.66) that produce **flat** SPC, pFR, and lag-CRP curves — even though the log-likelihood at the fit is 826 units better than at the C&Z 2025 defaults.

Per-position SPC (10-participant subset, 20 synthetic replications):

| position | observed | defaults | fit   |
|---------:|---------:|---------:|------:|
|        1 |    0.76  |    0.78  | 0.67  |
|        8 |    0.56  |    0.17  | 0.54  |
|       16 |    0.66  |    0.58  | 0.72  |

Recalls per list: observed = 10.47, defaults = 4.99, fit = 8.99.

**Interpretation.** The LL is dominated by stopping-rule terms (how many recalls per list), not by shape. Defaults produce correct shape but under-recall (~5 per list); the fit raises ε_d from 1.04 to 2.66, delays stopping, increases the number of recalls to ~9 — and in the process smears primacy and recency across the list. This is a genuine scientific result: **the likelihood-maximizing parameters on FRFR-category are not the behaviorally-realistic parameters.**

**Implication for paper.** The current paper claims "both models reproduce the canonical free-recall signatures" — false at the fit, true only at defaults. The paper must be rewritten to report this tension honestly; it is the most interesting methodological finding of the worked example.

### fig_behavioral caption is false

**Finding.** `paper/figs/source/fig_behavioral.png` shows ONLY observed (black) curves. The caption claims "observed (black) vs MS-TCM (red) and --standard-tcm (blue) posterior-predictive." and then asserts "Both models reproduce the U-shape", "Both models reproduce the end-of-list bias", "Both models reproduce the lag-+1 peak" — none of which is visible in the figure.

**Fix.** Delete the figure (redundant with fig_analyses per user direction) and rewrite the caption/text to describe what fig_analyses actually shows.

### fig_analyses shows MS-TCM failing to reproduce any of the canonical signatures

**Finding.** `paper/figs/source/fig_analyses.png` plots observed (black) vs MS-TCM (red dashed). MS-TCM is near-flat across all three panels: SPC at ~0.62 (no primacy bump, faint recency), pFR at ~0.05–0.07 (no primacy-end bias), lag-CRP at ~0.09–0.10 (no +1 peak, no forward asymmetry). The paper claims the opposite.

**Root cause.** fig_analyses was generated from the fit — and the fit is bad (see above). The figure is numerically accurate; the paper's text is wrong.

### fig_clustering under-represents the observed clustering gap

**Finding.** Observed clustering scores ≈ 0.67 / 0.67 / 0.63 (temporal / category / semantic). MS-TCM scores ≈ 0.52 / 0.53 / 0.52. The paper says "MS-TCM posterior-predictive tracks these in direction and rough magnitude." Direction ✓; magnitude ✗ (MS-TCM under-clusters by ~0.15 percentile-rank points on every dimension — half the observed above-chance signal).

### Primacy gradient discrepancy with C&Z 2025

**Finding.** C&Z 2025 explicitly **removed** the φ_s, φ_d primacy parameters from CMR in their hierarchical extension (p. 17: "thus removing two parameters φ_s and φ_d from the base CMR model"). Their hierarchical model gets primacy emergently from list-level boundary synchronization, not from a primacy-weighted M^CF.

**Our code.** `code/ms_tcm/hcmr.py` hardcodes a primacy gradient scaling `1 + phi·exp(-psi·t)` on M^IC Hebbian updates, with φ=1.5/ψ=0.5 under MS-TCM and φ=30/ψ=0.8 under --standard-tcm. This is **not** inherited from C&Z; it is a patch we added to make the behavioral regression pass.

**Fix.** The paper must document this. Either:
- justify re-adding φ_s/φ_d for FRFR-category (e.g., FRFR lists have only one list per participant-list trial, so C&Z's list-level emergent primacy mechanism can't operate); OR
- remove the hardcoded primacy gradient and accept that the Layer-1 shape assertions were passing for the wrong reason.

## Severity 2 — text that contradicts verified data

### Inflated "60 % same-category" claim

**Paper says** (§4 Interpretation, line 593): "given a just-recalled item from category A, roughly 60 % of next-recalls are also from category A."

**Actual.** Measured from FRFR-category recalled.parquet: 2332 same-category transitions out of 4725 consecutive pairs = **49.4 %**. Chance baseline (3/15 remaining) = 20 %.

**Fix.** Update to "roughly 49 %, well above the ~20 % chance baseline."

### Feature encoder description off-by-count

**Paper says** (§4, line 501): "colour (8 coarse bins of the RGB cube), and on-screen quadrant (4)".

**Actual** (`code/ms_tcm/features.py`): 4 bins per RGB channel × 3 channels = **12 RGB bins**; and 2 x-bins × 2 y-bins = **4 position bins** (the schema is additive not quadrant-multiplicative; the "4" happens to equal 2+2 but "quadrant" is wrong).

**Fix.** "colour (12 bins: 4 coarse bins of R, G, and B each), and on-screen position (4 bins: 2 for x, 2 for y)".

### C&Z 2025 attribution underplays what C&Z introduced

**Paper says** (Abstract, §2): "Cornell and Zhang (2025) recently extended CMR with a slower list-level hierarchical context..."

**Actual.** C&Z 2025 introduced THREE jointly-operating mechanisms (p. 5): (i) **Multiple Contexts** (item + list), (ii) **Boundary Synchronization** (c^item ← c^list at list boundaries, encoded in M^lists), and (iii) **Hierarchical Retrieval** (top-down search: when item-level search fails, retrieve list-level context and use it to reactivate item-level context at list boundaries). The paper treats C&Z as a single-mechanism extension and misses (iii). MS-TCM's λ is best described as a fourth mechanism (encoding-time storyline-return reinstatement), not as a tweak to one of C&Z's three.

**Fix.** Rewrite §1 (intro) + §2 (model) so C&Z's three mechanisms are explicitly enumerated before MS-TCM's λ is introduced.

## Severity 3 — figure regeneration

### fig_concept uses retired v1 symbols

**Finding.** Panel C of `paper/figs/source/fig_concept.png` labels contexts as "c_G" (global) and "c_S1, c_S2, c_S3" — these are v1 symbols that MS-TCM retired in feature 002. The paper body uses `c^item`, `c^story`. Figure must be regenerated in v6 notation.

Also: panel C's "frozen" label is technically inconsistent with v6 — inactive storylines *are* carried forward unchanged during drift, but the λ mechanism reinstates them at returns, so "frozen" is an over-simplification. A better label: "carried forward; reinstated at returns with strength λ".

### fig_model has LaTeX-escape artifact

**Finding.** The PNG title reads "MS-TCM v6 = Cornell \\& Zhang (2025) hierarchical CMR + storyline-return reinstatement" — the literal backslash-ampersand shows in the image. Regenerate with a clean string.

Also: the arrow from "feature f_i" to c^item is labeled β_enc, but β_enc is the drift rate applied in Eq 1 after the input vector c^IN is computed via Eq 1.5.1. Better label on the arrow: "c^IN_i" with β_enc on the drift loop.

### fig_experiment word colors are too pale

**Finding.** Some words (UNDERWEAR, KNIFE, PANTS) render in very low-contrast light colors against white background. Hard to read.

**Fix.** Either add a stroke/outline, or restrict the colour palette for the schematic to higher-saturation values while noting that the real experiment used arbitrary RGB.

### fig_behavioral: delete

Per user direction: redundant with fig_analyses.

### Behavioral figures need SEM error bars + lag-CRP forward/backward split

**User requirements.**
- Lag-CRP: exclude lag=0, show negative lags as ONE line and positive lags as a DIFFERENT line (two disconnected series, standard Kahana/Polyn convention).
- All behavioral curves: SEM error bars across participants.

**Fix.** Extend `code/analyses/spc.py`, `pfr.py`, `lag_crp.py` with `compute_*_per_participant` helpers, then regenerate `fig_analyses.py` with shaded SEM bands and split-lag CRP.

## Severity 4 — bibliographic issues

### local.bib HeusMann18 is wrong content

**Finding.** `paper/local.bib` defines:
```bibtex
@article{HeusMann18,
  author  = {A C Heusser and P C Fitzpatrick and J R Manning},
  title   = {Geometric models reveal behavioural and neural signatures of transforming experiences into memories},
  journal = {Nature Human Behaviour},
  volume  = {5},
  year    = {2021}
}
```

`paper/CDL-bibliography/cdl.bib` defines `HeusMann18` as the 2018 CCN proceedings paper — a different paper. The paper cites `HeusMann18` five times; from context all five intend the 2018 CCN paper (capturing geometric structure of episodic memories for naturalistic experiences). The local.bib override is wrong-content: it references a 2021 Nature HB paper with a wrong bibkey (should be e.g. `HeusFitzMann21`).

**Fix.** Remove the local.bib HeusMann18 entry so the canonical cdl.bib version wins. If any citation is meant for the 2021 NHB paper, add it under a new bibkey.

### local.bib duplicates PolyEtal09, SedeEtal08

**Finding.** `PolyEtal09` and `SedeEtal08` are identical in content between cdl.bib and local.bib. BibTeX will emit "repeated entry" warnings; cleanest fix is to remove from local.bib.

## Severity 5 — secondary text issues

### Relationship-to-existing-models section describes v6 in C&Z terms inconsistently

**Finding.** §5's "Relationship to existing models" paragraph on C&Z (line 669-676) says "hierarchical CMR has λ implicitly =0 (no storyline-return cache), whereas MS-TCM permits λ ∈ (0, 1]". This is correct at the math level but obscures a structural point: C&Z's model is over (item-level, list-level), with lists being trial-level structures; MS-TCM maps C&Z's "list" onto "storyline" which is within-list structure. The two models are structurally analogous but the mapping is "list ↔ storyline" not "same paradigm".

**Fix.** Add a paragraph explicitly describing the structural mapping: C&Z's list-level context ⇄ MS-TCM's storyline-level context; C&Z's M^lists ⇄ MS-TCM's M^SC; C&Z's boundary-sync between lists ⇄ MS-TCM's boundary-sync between storylines; and the λ mechanism activates only in the naturalistic multi-storyline-per-list case that C&Z did not study.

## Remediation plan (proposed, awaiting user sign-off)

1. **Diagnose and possibly fix the fit.** Why does the likelihood prefer ε_d=2.66 (sharp stopping, many recalls, flat curves) over ε_d=1.04 (correct shape, fewer recalls)? Option A: more restarts from a search grid rather than Gaussian perturbations of C&Z defaults. Option B: add a shape-aware regularizer to the objective. Option C: accept the result and report it as a scientific finding.

2. **Rewrite §2 to properly credit C&Z's three mechanisms** (Multiple Contexts, Boundary Synchronization, Hierarchical Retrieval); introduce MS-TCM's λ as a fourth, encoding-time mechanism.

3. **Regenerate all figures** with (a) v6 symbols, (b) cleaned LaTeX rendering, (c) SEM error bars across participants, (d) split-lag lag-CRP, (e) fig_behavioral deleted.

4. **Rewrite §4 (worked example) and §6 (discussion)** to report the real fit result honestly: the fit finds a regime with good LL but flattened behavioral curves; this reveals a tension between likelihood and shape that is interesting on its own terms.

5. **Fix the 60 % → 49 % number.**

6. **Fix the 12 RGB / 4 position feature description.**

7. **Fix bib duplicates and wrong-content HeusMann18.**

8. **Add a "primacy gradient" paragraph** to §2 acknowledging that MS-TCM re-adds C&Z's retired φ_s/φ_d, and justifying it (or removing it).

9. **Rebuild the paper and re-run `scripts/check_paper_consistency.py`.**

## Recommendation

The fit result is the load-bearing scientific issue. Fixing the ε_d-dominates-LL problem may require either a better optimizer (many restarts, basin-hopping, a grid search over ε_d specifically) or an explicit acknowledgement that on a paradigm with fixed recall counts, the likelihood is partly a nuisance-parameter problem and shape has to be enforced separately.

I recommend proceeding with the fit diagnosis first (option A: wider restart grid) before rewriting the paper, because the paper's story depends on whether a "good" fit exists or whether the tension is an unavoidable feature of this model×data combination.
