# Phase 0 Research: MS-TCM v6 Hierarchical CMR Rewrite

**Feature**: 002-ms-tcm-v6-hcmr
**Date**: 2026-04-23
**Input**: [plan.md](plan.md) and [spec.md](spec.md)

This document records the Phase 0 decisions for the v6 rewrite. The Technical Context in plan.md had no open NEEDS CLARIFICATION markers (all three were resolved in the `/speckit.clarify` pass — see spec.md §Clarifications). The research below documents (a) the constitution amendment required to recognize v6, (b) the parameter starting values inherited from Cornell & Zhang 2025, (c) the Tier 1 / Tier 2 performance strategy, (d) the behavioral regression methodology, and (e) the paper/doc migration approach.

## R0. Constitution amendment (MINOR bump 1.0.0 → 1.1.0)

**Decision**: Amend `.specify/memory/constitution.md` in the first commit of this feature branch.

**Rationale**: The current constitution (v1.0.0) identifies `notes/ms-tcm.pdf` as "the canonical specification". v6 replaces this with `notes/two_level_cmr_v6.pdf`. The constitution's own amendment procedure (§Governance > Versioning policy) defines a MINOR bump as "materially expanded guidance within an existing one" — pointing the canonical-spec reference at a successor document is exactly that. Principle III (Clarity) explicitly requires symbols to match the canonical spec on first use; if the canonical pointer stays at v1, every v6 symbol usage violates III.

**Changes**:

1. Replace `notes/ms-tcm.pdf` with `notes/two_level_cmr_v6.pdf` in the opening paragraph and in Principle I's §4.4 anchor example.
2. Replace the §4.4 numerical anchor language (Principle I) with the v6-appropriate anchor: "the `--standard-tcm` reduction (λ=0, one storyline, β_list drift) MUST reproduce the Cornell & Zhang 2025 free-recall phenomena per the Q2 Layer 1 shape-assertion policy; this is the required numerical-accuracy regression anchor for the CMR layer."
3. Update Principle III's symbol list (β_G, β_S, w_G, w_S, ρ_G, ρ_S, c_G, c_S, γ, λ) to v6 symbols (β_enc, β_story, γ_fc, k, λ, β_rec, ε_d, c^item, c^story).
4. Bump version footer to 1.1.0 with a Last Amended date of 2026-04-23 and a Sync Impact Report noting that the downstream templates (`plan-template.md`, `spec-template.md`, `tasks-template.md`) do not need edits.

**Alternatives considered**:

- **Leave the constitution stale, document the redirection in `CLAUDE.md` and `notes/v6_migration.md`**: Rejected. Creates two competing canonical pointers; a future feature's `/speckit.plan` run would not know which to trust. Principle III (Clarity) violated by construction.
- **Delete the §4.4 anchor language without replacement**: Rejected. Principle I (Accuracy) specifically requires a "hand-derived answer" anchor for numerical code. The `--standard-tcm` reduction is the v6-native replacement and must be named.

## R1. Parameter starting values (FR-009)

**Decision**: Inherit all starting values from Cornell & Zhang 2025 Table 1 and v6 §5.

**Values**:

| Parameter | Starting value | Source | Role |
|-|-|-|-|
| β_enc | 0.679 | C&Z 2025 Table 1 (labeled β_enc there too) | Item-level drift at encoding |
| β_story | 0.400 | C&Z 2025 Table 1 (there labeled β_list) | Storyline-level drift at encoding |
| γ_fc | 0.315 | C&Z 2025 Table 1 | Pre- vs. experimental context weighting |
| k | 6.50 | C&Z 2025 Table 1 | Softmax inverse temperature at retrieval |
| λ | 0.80 | v6 §5 | Storyline-return reinstatement strength (new) |
| β_rec | 0.326 | C&Z 2025 Table 1 | Retrieval drift (free-recall only) |
| ε_d | 1.04 | C&Z 2025 Table 1 | Stopping rule (free-recall only) |

**Rationale**: C&Z 2025 fit these values to Kahana et al. 2002 free-recall data and showed that they reproduce SPC, pFR, and lag-CRP at the qualitative shape level. Starting the optimizer from these values maximizes the probability that the MLE stays in a psychologically interpretable regime. γ_fc is treated as free in the fit (the FRFR-category multi-hot feature encoder may prefer a different mixture than one-hot word stimuli), but initialized at 0.315. λ has no direct precedent; 0.80 is the v6 §5 suggestion and reflects the prior that most of the returning storyline's context is re-instantiated.

**β_rein from C&Z Table 1 is not inherited** because v6 §5 explicitly replaces it with λ: "Cornell & Zhang's extension introduced 2 parameters (β_list, β_rein); one (β_story) is inherited directly, the other is replaced by λ, which is conceptually related but operates at encoding (on storyline return) rather than at retrieval."

**Alternatives considered**:

- **Uniform-random initialization**: Rejected. The optimizer is L-BFGS-B; random starts increase the chance of non-convergent bootstrap draws without any obvious benefit when published values exist.
- **Fit γ_fc at its C&Z 2025 value (0.315)**: Rejected. Different stimulus types (word lists vs. multi-hot feature vectors) likely want different pre-experimental weights; fixing it would leave a source of misfit unaccounted for.

## R2. Tier 1 performance strategy (FR-030)

**Decision**: Four targeted changes, no algorithmic restructuring.

1. **Vectorize `encode_features`**. Current code iterates via `pd.DataFrame.iterrows()` (line 206, ~100× slower than numpy broadcasting). Replace with a single pass that extracts columns as numpy arrays and scatters into the multi-hot layout. Expected speedup: ~30× on the encode step alone.

2. **Cache per-dataset features and pre-experimental matrix across likelihood evaluations**. Current `dataset_log_likelihood` re-runs `model.encode(dataset)` (likelihood.py line 84), which re-runs `encode_features` on the full dataframe every call. With ~5000 likelihood evaluations per fit (5 restarts × 1000 bootstraps), this is the dominant cost. Introduce a `DatasetFeatureCache` that holds the feature matrix and M^FC_pre, keyed by `id(dataset)`, invalidated only by a new Dataset instance.

3. **Vectorize the per-list encoding recurrence**. Current `encode` (model.py lines 85-115) runs a Python `for t in range(W)` with a nested `for other_cat in categories`. With W=16 and 4 categories per list, this is ~64 inner iterations per list, ~30k per dataset. Replace with a matrix formulation: given fixed β_enc and β_story, each storyline's context trajectory is the solution to a scalar recurrence that can be computed in a single np.einsum or np.matmul call over (W+1, d). Expected speedup: ~10-20× on encoding.

4. **Parallelize the bootstrap loop via `multiprocessing.Pool`**. Current bootstrap.py line 99 runs draws serially. Each draw is independent (they only share the read-only input dataset), making it embarrassingly parallel. `multiprocessing.Pool(n_processes=os.cpu_count())` with the draws as the unit of work should give ~cpu_count× speedup (typically 8-16× on a modern laptop, 2× on CI hardware). Special care: pickling a Dataset across the process boundary is expensive; send only the numpy arrays and rebuild on the worker. Workers receive a seed derived from the master's `SeedSequence` so determinism is preserved (the spec-required FR-018 / FR-034 constraint).

**Expected combined speedup**: conservatively 5× (on CI 2-core) to 20-30× (on a developer laptop). Tier 1 target of < 120 s on CI is comfortable.

**Rationale**: These four are the highest-leverage changes identified during the audit in the previous session. They are pure Python, introduce no new dependencies, and preserve the public API semantics (FR-034). All four are reversible (each is a local rewrite).

**Alternatives considered**:

- **Rewrite the inner loop in Cython**: Rejected at Tier 1 (violates "no new compilation step" goal). Reconsidered as Tier 3 only if JAX + multiprocessing fall short.
- **Replace scipy L-BFGS-B with a custom gradient descent**: Rejected. L-BFGS-B is not the bottleneck; the inner likelihood evaluation is. Keep scipy until Tier 2, when JAX autodiff provides the gradients anyway.

## R3. Tier 2 performance strategy (FR-032)

**Decision**: Optional JAX backend opted in via `MS_TCM_BACKEND=jax`; autodiff replaces finite-difference gradients; `vmap` over lists and participants.

**Rationale**: JAX's JIT compiler fuses the encoding recurrence, retrieval softmax, and log-likelihood summation into a single XLA kernel, eliminating Python-level overhead entirely. `vmap` over the (participant, list) axis gives effective SIMD parallelism within a single process; combined with multiprocessing at the bootstrap level, this gives a two-layer parallelism model. L-BFGS-B is replaced by a JAX-native optimizer (Optax's `lbfgs`, or `scipy.optimize.minimize` with `jac=jax.grad(...)`). With analytic gradients, each optimizer iteration saves ~7 likelihood evaluations (the finite-difference count for 7 parameters), a ~7× speedup on top of the JIT gains.

**Tier 2 target of < 30 s**: achieved by combining (a) JIT-fused inner loop (~5× over Tier 1 numpy), (b) analytic gradients (~7× on optimizer iterations), and (c) vmap over lists/participants (~2-4× on larger datasets). Net ~20× over Tier 1.

**Dtype policy (Q3 resolution)**: The `MS_TCM_JAX_DTYPE` env var / `--jax-dtype` CLI flag selects float64 (default, 1e-10 tolerance) or float32 (opt-in, 1e-8 tolerance). float64 is enforced in JAX via the runtime config `jax.config.update("jax_enable_x64", True)` at import time.

**Alternatives considered**:

- **Make JAX the default**: Rejected. Adds ~800MB to the install footprint (jax + jaxlib + optax + their CUDA/Metal dependencies), which is user-hostile for anyone who only wants to run `ms-tcm validate`. Keeping it behind an extras group (`pip install ms-tcm[jax]`) preserves the lightweight default install.
- **PyTorch instead of JAX**: Rejected. PyTorch's eager mode is slower on this workload than JAX's JIT; TorchScript compilation has weaker autodiff support than `jax.grad`. JAX wins on every axis for this kind of fixed-shape scientific computation.

## R4. Behavioral regression methodology (Q2, FR-020, FR-021)

**Decision**: Two-layer test in `code/tests/test_behavioral_regression.py`.

**Layer 1 — Qualitative shape assertions** (run on every CI invocation, ~10 s):

After fitting MS-TCM on a 3-participant subset of FRFR-category with `--n-bootstraps 5 --n-restarts 2` (a cheap smoke-fit), simulate recalls via `sample_recalls`, compute SPC / pFR / lag-CRP, and assert:

- `spc[0] > spc[W/2]` (primacy limb detectable)
- `spc[W-1] > spc[W/2]` (recency limb detectable)
- `argmax(pfr) > W/2` (first recall biased toward end of list)
- `argmax(crp[non_zero_lags]) == +1` (peak at lag +1)
- `crp[+1] > crp[-1]` (forward asymmetry)
- `crp[+1] > crp[+2]` (monotone falloff forward)

The 3-participant smoke fit is fast enough to run on every commit but exercises the whole pipeline (encode → fit → sample → analyze). It is not quantitatively meaningful — Layer 2 handles that.

**Layer 2 — Relative-fit assertion against FRFR-category empirical curves** (run on every CI invocation, ~90 s):

Before CI starts, `scripts/build_reference_curves.py` computes empirical SPC / pFR / lag-CRP from `data/raw/frfr_category/recalled.parquet` and writes three Parquet files under `data/processed/reference_curves/` with a companion `manifest.json` holding SHA-256s. The committed reference curves are self-verifying.

The test then runs a medium fit (`--n-bootstraps 50 --n-restarts 3`, ~60 s on Tier 1), simulates recalls, computes the three curves, and asserts:

1. For each of SPC, pFR, lag-CRP: the per-bin relative error `|sim[i] - emp[i]| / max(emp[i], 1e-6)` is ≤ 0.20 (20 %).
2. Simultaneously, the test fits the `--standard-tcm` reduction, simulates its recalls, computes the same three curves, and asserts that the MS-TCM total absolute error is ≤ the standard-TCM total absolute error on each curve (MS-TCM no-worse-than standard-TCM).

**Rationale**: Layer 1 catches gross regressions (the model no longer produces a U-shaped SPC) quickly. Layer 2 provides the scientific bar (the model fits the data it's supposed to fit and does no worse than its own reduction). Both layers live in one test file, and any failure identifies *which curve* and *which layer* failed for fast triage.

**Alternatives considered**:

- **Match against C&Z 2025 Figure 2 digitized values**: Rejected. Kahana et al. 2002 stimuli (10 words/list, uncategorized) differ materially from FRFR-category (16 words/list, 4 categories); a C&Z-calibrated target would fail for the wrong reason.
- **Match against Kahana et al. 2002 Figure 1 directly**: Rejected for the same reason as above.
- **Use a Chi-squared goodness-of-fit test**: Rejected. Chi-squared requires per-bin counts, not per-bin probabilities; translating the simulated probabilities into counts requires choosing a simulation N that's arbitrary. Per-bin relative error is both simpler and more transparent.

## R5. Paper and documentation migration (FR-040 through FR-049)

**Decision**: In-place rewrite of `paper/main.tex` §3 and §4; new subsection for M^FC_pre; local `paper/local.bib` for the C&Z 2025 entry (submodule is read-only in this feature); `CLAUDE.md` and the three READMEs updated to v6 notation; `notes/v6_migration.md` authored as the single source of truth for the v1→v6 symbol mapping.

**Key edits**:

1. **paper/main.tex**:
    - §3 "Multi-Stream Temporal Context Model" → rewritten as "Hierarchical Context Model of Narrative Memory", following v6 §1–2. Two context levels, M^IC / M^SC associative matrices, boundary sync, λ reinstatement. Figure 1 (concept diagram) regenerated from `code/figures/make_fig_model.py`.
    - §4 "Deriving the equivalence" → rewritten to the v6 narrative: grouped vs. bridge similarity derived from the *item-level context* after boundary sync + the *storyline context* reinstated at storyline returns with weight λ. §4.4 numerical anchor removed.
    - Add §1.5 equivalent "The pre-experimental matrix" (v6 §1.5) covering identity vs. USE-embedding choice for M^FC_pre.
    - Add one explicit sentence in §3 or §Methods: "MS-TCM adds exactly one mechanism to Cornell & Zhang's (2025) hierarchical CMR: storyline-return reinstatement at encoding-time with strength λ (§2.4)."
    - Methods section updated with the new parameter inventory (Table listing β_enc, β_story, γ_fc, k, λ, β_rec, ε_d with C&Z 2025 Table 1 citations).
    - Figure 2 (behavioral) added: SPC / pFR / lag-CRP for FRFR-category human data vs. MS-TCM vs. standard-TCM.

2. **paper/local.bib** (new):

   ```bibtex
   @article{CornellZhang2025,
     author  = {Cornell, Charlotte A. and Zhang, Qiong},
     title   = {Hierarchical Context Guides Human Memory Search},
     journal = {Psychological Review},
     year    = {2025},
     note    = {preprint: \url{notes/CornZhan25.pdf}}
   }
   ```

   `main.tex` adds `\bibliography{CDL-bibliography/cdl,local}` to include both bibs. If during implementation the submodule turns out to be editable in this feature's window, the entry migrates to `CDL-bibliography/cdl.bib` and `local.bib` is deleted; the choice is left to the implementer at the commit boundary.

3. **CLAUDE.md**: Replace the "Project" section's canonical-spec reference (`notes/ms-tcm.pdf` → `notes/two_level_cmr_v6.pdf`). Replace the v1 parameter-regimes list (β_G, β_S, w_G, w_S, γ, λ) with v6 symbols (β_enc, β_story, γ_fc, k, λ, β_rec, ε_d). Remove the §4.4 numerical-sanity-check paragraph (replace with a pointer to `test_hcmr_standard_tcm.py` as the accuracy anchor).

4. **README.md, code/README.md, data/README.md**: Update CLI examples to v6 parameter names; add a one-liner about the Tier 1 performance claim ("fits in under 2 minutes on a laptop"); point at `specs/002-ms-tcm-v6-hcmr/` as the active feature spec.

5. **specs/001-ms-tcm-impl/plan.md**: Add a one-line "SUPERSEDED by specs/002-ms-tcm-v6-hcmr/plan.md — v1 MS-TCM math retired; see notes/v6_migration.md" at the top.

6. **notes/v6_migration.md** (new):
    - Symbol mapping table: v1 β_G ↔ v6 β_enc (with caveats — see note below); v1 β_S ↔ v6 β_story; v1 w_G / w_S constraint → REMOVED; v1 γ (resumption, §5.1) ↔ v6 λ (Eq 6) with a conceptual-equivalence caveat (v1 γ was additive; v6 λ is a linear blend); v1 w_G^ret / w_S^ret → REMOVED (v6 retrieval is instruction-blind).
    - Deletion manifest: list of removed files and tests (FR-013 + FR-048).
    - Rationale section: 3–5 paragraphs summarizing why v6 was adopted (standard CMR machinery produces SPC/pFR/lag-CRP; v1 didn't).

**`scripts/check_paper_consistency.py`** (FR-049): greps the compiled `main.tex` body (excluding bibliography and the migration note, if any) for the literal strings `\\beta_G`, `\\beta_S`, `w_G`, `w_S`, `0.866`, `0.806`, `0.80532`, `frozen storyline`, `w_{\\mathrm{global}} + w_{\\mathrm{storyline}}`. Any match exits non-zero with a file/line pointer.

**Rationale**: In-place rewrites with a single-source migration document minimize the chance that v1 language re-enters the paper during later edits. The `check_paper_consistency.py` script provides a CI-level gate against documentation rot.

**Alternatives considered**:

- **Separate v1 and v6 papers**: Rejected. There's only one paper; a rewrite is easier than maintaining two.
- **Leave the bibliography submodule alone and not cite C&Z 2025**: Rejected. FR-041 requires the citation. A local `.bib` is a clean fallback.

## R6. Data artifacts and regeneration scripts

**Decision**:

- `data/processed/reference_curves/frfr_category_spc.parquet`: empirical SPC from FRFR-category.
- `data/processed/reference_curves/frfr_category_pfr.parquet`: empirical pFR.
- `data/processed/reference_curves/frfr_category_lag_crp.parquet`: empirical lag-CRP.
- `data/processed/reference_curves/manifest.json`: SHA-256 hashes, row counts, source (FRFR-category manifest hash).
- `data/processed/benchmarks/benchmark_log.csv`: append-only history of benchmark runs (tier, wall_clock_seconds, peak_memory_mb, n_bootstraps, n_restarts, git_sha, platform, timestamp).

**Regeneration**: `scripts/build_reference_curves.py data/raw/frfr_category --out data/processed/reference_curves/` is idempotent (same input → same output bytes).

## R7. Dependency acquisition

**Decision**: `jax`, `jaxlib`, `optax`, `pytest-xdist` are added only to `[project.optional-dependencies].jax` and `[project.optional-dependencies].dev` respectively. The default `pip install -e .` install is unchanged.

**Rationale**: Most users will install only the Tier 1 path. Tier 2 is opt-in both for the install surface and the runtime switch.

## Summary of Phase 0 artifacts

- Constitution amendment scripted (R0) — lands in the first commit.
- Parameter starting values pinned to C&Z 2025 Table 1 + v6 §5 (R1).
- Tier 1 performance plan has four concrete code changes (R2) — no new deps.
- Tier 2 performance plan is optional, JAX-based, dual-dtype (R3) — adds an extras group.
- Behavioral regression is two-layer: qualitative shape + relative-fit (R4).
- Paper/doc migration is in-place rewrite + `local.bib` + `v6_migration.md` (R5).
- New data artifacts: reference curves + benchmark log (R6).
- Dependencies: default install unchanged; Tier 2 is `pip install ms-tcm[jax]` (R7).

Ready for Phase 1.
