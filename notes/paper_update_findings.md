# Paper Update Findings (2026-04-26)

Audit of `paper/main.tex` and `paper/supplement.tex` against the current
state of the project.

**STATUS UPDATE (2026-04-26 post-consolidation):** Constitution II
consolidation is complete. The MS-TCM code now lives at a single
canonical path (`code/ms_tcm/_likelihood_core_mstcm.py`) and the paper
has been rewritten to describe MS-TCM as ONE model with a single set
of mechanisms. The "Iteration 1 / Iteration 2 / v1 / v2" framing has
been removed from both `main.tex` and `supplement.tex`; design
history lives only in `notes/mstcm_architecture_log.md`.

## Resolution status of prior audit items

1. **"MS-TCM adds exactly one mechanism (λ) to C&Z 2025."**
   - **RESOLVED.** Abstract and §2 of `main.tex` now describe MS-TCM
     as adding four mechanisms over C&Z: (a) per-storyline contexts
     plus a separate global cross-storyline context (drifting at every
     encoding step at rate $\beta_\mathrm{enc}^G$); (b) per-storyline
     associative matrices ($M^{FC}_s$, $M^{CF}_s$) alongside the
     global ($M^{FC}_G$, $M^{CF}_G$); (c) λ storyline-return
     reinstatement at encoding (v6 Eq 6); (d) a τ-mixture between a
     recency-driven global recall route and a strict hierarchical
     route in which the global context cues a storyline and the
     storyline cues its events. The relationship-to-existing-models
     section was updated correspondingly.

2. **Empirical FRFR-category fit description.**
   - **PARTIALLY RESOLVED.** The C&Z 2025 fit numbers
     (LL = -10699.82, full parameter MLE) are reported in §4 from
     `data/processed/fits/cz_frfr/fit_summary.json` (still accurate,
     same code path as before consolidation). The MS-TCM
     `data/processed/fits/mstcm_frfr/fit_summary.json` exists but is
     STALE — its mtime is Apr 26 00:37, before the consolidation
     commit `b9e9c72` at Apr 26 14:59. It corresponds to the
     pre-consolidation MS-TCM (no β_enc_global, single retrieval
     route plus τ-mixture, β_list collapsed to ~0). The paper does
     NOT report these stale numbers as the current MS-TCM fit;
     instead §4 notes that "the MS-TCM fit on the consolidated
     implementation is being re-run" and §8 (Discussion) defers
     ΔAIC reporting to once that fit completes.
   - TODO for the human: re-run MS-TCM MLE on the consolidated
     implementation (`scripts/run_fit.py` or equivalent) and update
     §4 with the resulting parameters and ΔAIC. The
     fig_analyses behavioral overlay can stay as-is qualitatively
     (the per-storyline + λ + strict-hierarchical mechanisms are
     present in both pre- and post-consolidation runs); only the
     numerical LL and AIC values change.

3. **"--standard-tcm reduction" framing.**
   - **RESOLVED.** §6 (Relationship to existing models) now states
     that MS-TCM reduces to standard CMR when $K=1$, $\tau=0$, and
     $\lambda = 0$, and reduces to C&Z hierarchical CMR when $K=1$
     (with the global and per-storyline branches collapsing onto a
     single context). `--standard-tcm` is referenced in
     supplement §S3 (primacy gradient) and §S4 (numerical anchor)
     as the implementation switch that disables the hierarchical
     layer for the C&Z baseline; it is NOT framed as the
     model-comparison comparator (the C&Z fit at
     `data/processed/fits/cz_frfr/` is).

4. **τ-mixture initiation is missing entirely.**
   - **RESOLVED.** §2.4 of `main.tex` (Retrieval) now describes the
     τ-mixture explicitly, with route α (recency-driven global
     recall using $M^{CF}_G$) and route β (strict hierarchical
     recall: global cues storyline through $M^\mathrm{lists}_G$,
     storyline cues events through $M^{CF}_s$, immediately-preceding
     storyline excluded on global re-selection). Equations
     `eq:softmax` and `eq:tau-storyline-init` are the relevant
     equations. The mixture is selected once at recall onset and
     used for the entire recall sequence (matches the implementation
     in `_likelihood_core_mstcm.py`).

5. **Behavioral diagnostics missing.**
   - **RESOLVED.** §4.1 of `main.tex` (Behavioural diagnostics)
     reports both findings: the lag-CRP composition shift (within-cat
     lag-CRP at +1 is essentially identical between halves at ~0.55;
     across-cat at ~0.21; early lists are 57% within-category vs
     38% in late lists) and the late-list scallop (amplitude 0.074,
     permutation p=0.001, fully explained by category-organized
     recall at r=0.999, output-position chunking rejected). Both
     findings now flow into the model-implications paragraph,
     which connects them to the strict hierarchical route's
     within-storyline clustering.

6. **Figure references.**
   - **RESOLVED.** `fig_analyses` is now cited as the primary
     behavioral comparison figure in §4 (with caption updated to
     reflect MS-TCM rather than "Iteration 1"). `fig_concept`,
     `fig_model`, `fig_experiment` continue to be cited as before.
     `fig_clustering` and `fig_behavioral` are no longer cited in
     the paper (the diagnostic plots referenced in
     `notes/clustering_analysis_figures/` and
     `notes/late_list_scallop_figures/` are pointed to from the
     diagnostics section).

7. **Constant numerical values from v6 spec.**
   - **RESOLVED.** §4 now lists initialization values for $\lambda$
     ($0.80$ per `notes/two_level_cmr_v6.pdf` §5), $\tau$ ($0.5$),
     and $\beta_\mathrm{enc}^G$ ($0.400$, the C&Z $\beta_\mathrm{list}$
     rate as initialization; the parameter is fit from the data).
     Fitted values will be added once the consolidated MS-TCM fit
     completes.

8. **Supplement S0/S2/S3/S4 reference v1 → v6 only.**
   - **RESOLVED.** Supplement §S0 has been rewritten to record the
     architectural lineage as a single arc (early v1 → v6 spec →
     mechanisms added during the FRFR-category worked example) that
     converged on the present specification. The "Iteration 2"
     framing is gone; §S0 instead points to
     `notes/mstcm_architecture_log.md` for the design rationale of
     each mechanism. Supplement §S4 (numerical-accuracy anchors)
     was updated to enumerate the new mechanisms (per-storyline
     contexts, global cross-storyline context, per-storyline
     matrices, λ, τ-mixture between recency and strict-hierarchical
     routes) rather than listing only λ.

## Constitution II violation resolution

The parallel `_likelihood_core_mstcm.py` (formerly Iteration 1) and
`_likelihood_core_mstcm_v2.py` (formerly Iteration 2) files have been
consolidated: as of commit `b9e9c72`, the v2 implementation REPLACED
the v1 code in place at the canonical path
`code/ms_tcm/_likelihood_core_mstcm.py`. The paper now reflects this:
the model is described once, with a single mechanism inventory, and
matches the consolidated code.

Items left as TODOs for the human:
- Re-run MS-TCM MLE on the consolidated implementation and update §4
  with current parameter values and ΔAIC. (The fit_summary.json at
  `data/processed/fits/mstcm_frfr/` is stale.)
- Re-run §S2 grid-scan diagnostic against the consolidated MS-TCM
  $(\beta_\mathrm{enc}, \gamma_\mathrm{fc})$ scan. (Marked as a TeX
  comment in supplement.tex; the qualitative pattern stands but the
  nat ranges are from an earlier parameterization.)

## Compile status (post-consolidation)

- `paper/main.pdf` regenerated successfully (273 KB, fresh).
- `paper/supplement.pdf` regenerated successfully (94 KB, fresh).
- No LaTeX errors. One overfull-hbox warning in supplement §S3
  (Primacy gradient paragraph) — cosmetic, not new.
- `grep -ci "iteration [12]\|\bv[12]\b" main.tex supplement.tex`
  returns 0 in both files.

## Bibliography status

All needed keys exist. CDL keys: CornellZhang2025, PolyEtal09,
HowaKaha02a, HeusMann18, MortPoly16, XuEtAl2024, SedeEtal08,
ManningEtAl2023FRFR, XuDuncanManning, ZacSpeSwaBraRey07, Wic70,
EzzaDav11, Kahana96, KahaEtal08, BurAnd04, GodBad75 (all in local.bib);
HeusMann18, ZadbEtal17, BaldEtal17, BaldEtal18, ZwaaRadv98, ClewDava17,
HowaKaha99, BjorWhit74, PolyKaha08 (in cdl.bib). No new bibliography
work is needed.

## Accurate / OK content (kept across the rewrite)

- §1 lit review: factually fine; only minor edits to clarify
  MS-TCM's relation to C&Z hierarchical CMR.
- §3 (analytic derivation of grouped-bridge equivalence): the
  analysis is unchanged (the encoding-time storyline-return
  reinstatement mechanism is preserved verbatim across the
  consolidation).
- §5 (predictions): all five predictions still hold; small framing
  edits to remove the "without τ" / "Iteration 2" framing.
- §6 (relationship to existing models): updated to enumerate the
  four mechanisms cleanly; the underlying claims are unchanged.
- §7 (caveats): unchanged. The FRFR-category limitation, the
  storyline-analog-is-thin caveat, and the λ-identifiability
  caveat all stand.
- Supplement §S1 (off-by-one fix), §S3 (norm-preserving drift,
  softmax gain, primacy, no-repeats): factually accurate, kept.
