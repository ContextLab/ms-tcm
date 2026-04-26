# Paper Update Findings (2026-04-26)

Audit of `paper/main.tex` and `paper/supplement.tex` against the current
state of the project (C&Z 2025 verified baseline; MS-TCM Iteration 1
committed and fitted; MS-TCM Iteration 2 implementation in progress).

## Stale claims (need fixing)

1. **"MS-TCM adds exactly one mechanism (λ) to C&Z 2025."**
   - Where: abstract; §2 first paragraph; §6 (relation-to-existing); §7
     (discussion).
   - Quote (main.tex line 63-64): *"MS-TCM that extends Cornell and
     Zhang's (2025) hierarchical CMR with exactly one additional
     mechanism: storyline-return reinstatement at encoding with strength
     λ."*
   - Status: **stale.** MS-TCM Iteration 1 (committed) adds THREE
     mechanisms over C&Z: (a) per-storyline contexts (one per category),
     (b) λ storyline-return reinstatement at encoding, (c) τ
     storyline-initiation mixture at recall onset (NEW; not in v6 spec).
     Iteration 2 (in progress) adds (d) a separate global cross-storyline
     context with its own drift β_enc^G, (e) per-storyline associative
     matrices M^FC_s, M^CF_s, (f) strict hierarchical retrieval (route β:
     global → storyline → events; storyline-stop returns to global with
     immediately-preceding storyline excluded).

2. **Empirical FRFR-category fit description.**
   - Where: §4 (worked example); supplement §S2.
   - Currently the paper does not report the actual fitted parameters
     or LL deltas. The actual fit values are in
     `data/processed/fits/cz_frfr/fit_summary.json` (LL = -10699.82) and
     `data/processed/fits/mstcm_frfr/fit_summary.json`
     (LL = -10502.16, ΔAIC = -391.32). MLE: τ ≈ 0.408,
     λ_reinstate fixed near 0.5, β_list collapses to ~0.
   - Status: **stale / underspecified.** The paper should report the
     actual numbers and discuss the τ ≈ 0.41 finding.

3. **"--standard-tcm reduction" framing.**
   - Where: §2 (Relation to standard CMR), §4 (Optimizer/CIs), §6.
   - Quote (main.tex line 357-363): *"When λ = 0 and the full dataset
     is treated as a single storyline... MS-TCM reduces to standard CMR
     [PolyEtal09]. We use this second reduction in §4 as the
     `--standard-tcm` baseline for model comparison."*
   - Status: **stale.** The current model comparison baseline is C&Z
     2025 hierarchical free recall (already two-level), not standard CMR.
     The C&Z fit at `data/processed/fits/cz_frfr/` is the apples-to-apples
     baseline; MS-TCM Iteration 1 is compared to it directly.
     `--standard-tcm` is a numerical-accuracy anchor (verified in
     `code/tests/test_hcmr_standard_tcm.py`) but is not the reported
     comparator.

4. **τ-mixture initiation is missing entirely.**
   - Where: §2 (Retrieval) — line 326-351.
   - Quote (main.tex line 326-329): *"Retrieval: the standard CMR route,
     instruction-blind. At retrieval, MS-TCM inherits the standard CMR
     readout [PolyEtal09, CornellZhang2025]."*
   - Status: **stale.** The committed MS-TCM (Iteration 1) has a τ
     storyline-initiation mixture at recall onset. The current paper
     writes about retrieval as if MS-TCM inherits C&Z verbatim. With τ,
     the recall-onset cue is, with prob τ, sampled via a storyline
     hierarchy (storyline ŝ ∝ softmax(k · M^SC · c^list_end), then
     reactivate item context to c^story_ŝ); with prob 1-τ, the standard
     end-of-list cue.

5. **Behavioral diagnostics missing.**
   - Currently absent. The two empirical findings documented in
     `notes/clustering_analysis.md` and
     `notes/late_list_scallop_analysis.md` are not mentioned anywhere.
   - Status: **need to add.** These motivate Iteration 2's strict
     hierarchical retrieval. Specifically:
     - The lag-CRP early/late difference is a COMPOSITION SHIFT, not a
       within-condition mechanism difference (within-cat lag-CRP at +1
       ≈ 0.55 in both halves; across-cat ≈ 0.21 in both halves; early
       lists have 57% within-cat transitions vs 38% in late).
     - The late-list scallop is statistically real (p = 0.001) and
       FULLY explained (r = 0.999) by category-organized recall — a
       within-category mini-SPC averaged over scattered category-onset
       positions, NOT output-position chunking.

6. **Figure references.**
   - The paper currently references `fig_concept`, `fig_model`,
     `fig_experiment`, `fig_behavioral`, `fig_clustering`. The
     newer/canonical analysis figure `fig_analyses.pdf` (early/late
     overlay of observed FRFR + C&Z + MS-TCM v1 by row × pFR/lag-CRP/SPC
     columns) is NOT cited.
   - Status: **fix.** The paper should cite `fig_analyses` as the
     primary behavioral comparison figure.

7. **Constant numerical values from v6 spec that no longer apply.**
   - Where: §4 (Optimizer): main.tex line 519 lists "λ = 0.80
     initialization (notes/two_level_cmr_v6.pdf §5)".
   - Status: **partially stale.** The committed fit found λ ≈ 0.5
     (fixed in the optimizer setup), τ ≈ 0.41. The starting values can
     stay but the fitted values should appear.

8. **Supplement S0/S2/S3/S4 reference v1 → v6 only.**
   - Where: supplement.tex §S0 ("v1 → v6 migration"), §S4
     ("Numerical-accuracy anchor: the standard-CMR reduction").
   - Status: still accurate as a historical note but should be updated
     to (a) explicitly state that C&Z 2025 is now an independently
     verified implementation and is the comparator, not standard CMR,
     and (b) note that MS-TCM has evolved to include τ and is moving to
     Iteration 2 (per-storyline + global + strict hierarchical retrieval).

## Accurate / OK content (keep)

- §1 ("recency / contiguity / TCM/CMR / hierarchical CMR" lit review):
  factually fine; only minor edits to flag that MS-TCM extends C&Z
  hierarchical CMR.
- §3 (analytic derivation of grouped-bridge equivalence): the analysis
  is still valid for the encoding-time storyline-return reinstatement
  mechanism per v6 Eq 6, which is preserved across all current
  iterations of MS-TCM. It does not need to be rewritten, only the
  framing should clarify that this is one of several mechanisms in the
  current MS-TCM.
- §5 (predictions): all five predictions are still valid — they hinge
  on storyline-context mechanics, which are present in all iterations.
- §6 (relationship to existing models): largely fine; just need to
  acknowledge that MS-TCM now adds more than one mechanism over C&Z.
- §7 (caveats): still appropriate; the FRFR-category limitation and the
  pending XuDuncanManning data hold.
- Supplement §S1 (off-by-one fix), §S3 (norm-preserving drift, softmax
  gain, primacy, no-repeats): factually accurate, keep.

## Recommended structure for updated paper

1. Abstract: announce C&Z 2025 baseline; MS-TCM adds *three* mechanisms
   (per-storyline contexts; λ; τ); Iteration 2 *strict hierarchical
   recall with separate global cross-storyline context* in progress.
2. §2 (Model): describe the architecture honestly. Three subsections:
   (a) C&Z 2025 baseline; (b) MS-TCM v1 mechanisms (per-storyline + λ
   + τ); (c) MS-TCM Iteration 2 (per-storyline + global + strict
   hierarchical retrieval) — flag fit results pending.
3. §3 (Equivalence derivation): keep the λ analytic argument; clarify
   it operates within the broader MS-TCM architecture.
4. §4 (Worked example): report actual C&Z and MS-TCM v1 LLs and ΔAIC;
   show fig_analyses overlay; report behavioral diagnostics
   (composition shift, scallop = category-organized recall) as
   motivation for what Iteration 2 must capture.
5. §5 (Predictions): keep.
6. §6 (Relationship to existing models): keep; add note on how MS-TCM
   relates to category-clustering models.
7. §7 (Discussion): update to reflect honest current state.

## Bibliography status

All needed keys exist. CDL keys: CornellZhang2025, PolyEtal09,
HowaKaha02a, HeusMann18, MortPoly16, XuEtAl2024, SedeEtal08,
ManningEtAl2023FRFR, XuDuncanManning, ZacSpeSwaBraRey07, Wic70,
EzzaDav11, Kahana96, KahaEtal08, BurAnd04, GodBad75 (all in local.bib);
HeusMann18, ZadbEtal17, BaldEtal17, BaldEtal18, ZwaaRadv98, ClewDava17,
HowaKaha99, BjorWhit74, PolyKaha08 (in cdl.bib). No new bibliography
work is needed.

## Sources of inconsistency I had to resolve

- v6 spec (`two_level_cmr_v6.pdf`) describes MS-TCM as "C&Z + λ only."
  But the committed code in
  `code/ms_tcm/_likelihood_core_mstcm.py` adds τ-initiation as a NEW
  mechanism (documented in `mstcm_architecture_log.md` Iteration 1).
  → I trust the architecture log + committed code over the v6 spec for
  describing what the current MS-TCM does. The v6 spec is the v6 spec;
  the project has moved past it.
- The architecture log says β_list collapsed to ~0 in the MS-TCM v1
  fit ("with τ active, the list-level context becomes redundant —
  parameter substitution"). The fit JSON confirms (β_list ≈ 4.45e-7).
  This is worth surfacing in the worked example as a diagnostic.
- `_likelihood_core_mstcm_v2.py` exists and is partially implemented;
  its core hyperparameters bundle exposes `beta_enc_global`. Not yet
  fitted. The paper should describe its architecture and note "fit
  pending."
