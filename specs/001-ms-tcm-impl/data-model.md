# Data Model: MS-TCM Datasets, Model State, and Fit Results

**Feature**: `001-ms-tcm-impl`
**Date**: 2026-04-21 (rewritten from the 2026-04-20 synthetic-dataset framing to match spec §FR-008 … FR-021 and the FRFR-category worked example)

This document defines the in-memory types, the on-disk schema, the multi-hot feature encoder, the encoding/retrieval state machine, and the fit-result shape. Symbols and equations match `notes/ms-tcm.pdf` exactly (Constitution Principle III).

## 1. Entities

### 1.1 Participant

One human participant in an FRFR session (or an equivalent future free-recall dataset).

| Field | Type | Constraints |
|-|-|-|
| `participant` | `int64` | Dense 0..P−1 within a dataset. Downstream bootstrap resampling groups at this level. |

### 1.2 List

One study-then-recall unit. A session contains 16 lists; 16 words are presented per list; the participant is then prompted to recall in any order. Early lists (`list < 8`) are sorted by category; late lists (`list ≥ 8`) are randomly ordered.

| Field | Type | Constraints |
|-|-|-|
| `list` | `int64` | Dense 0..L−1 within a participant (L = 16 for FRFR-category). |
| `list_group` | `string` | Derived: `"early"` if `list < 8` else `"late"`. |

Context vectors (global and every storyline) reset to zero at every list boundary (see §4.1).

### 1.3 Presented word (event)

One studied item. The atomic encoding unit.

| Field | Type | Constraints |
|-|-|-|
| `participant` | `int64` | References `Participant.participant`. |
| `list` | `int64` | References `List.list`. |
| `serial_position` | `int64` | 1..W (W = 16 for FRFR-category). 1-based to match the paper's convention. |
| `word` | `string` | Uppercase English noun; length in `[3, 12]`. |
| `category` | `string` | One of 15 semantic categories in the FRFR wordpool. **Also the storyline label** within its list (exactly 4 unique categories per list, so K = 4). |
| `size` | `string` | `"small"` or `"large"`. |
| `first_letter` | `string` | Single uppercase letter A–Z. |
| `word_length` | `int64` | Length of `word` in letters; 3..12. |
| `color_r`, `color_g`, `color_b` | `int64` | RGB in [0, 254]; -1 when reduced. |
| `pos_x`, `pos_y` | `float64` | Display coordinates as fractions of the viewable area; NaN when reduced. |
| `list_group` | `string` | Derived; see §1.2. |

### 1.4 Recalled word

One recall output.

| Field | Type | Constraints |
|-|-|-|
| `participant` | `int64` | References `Participant.participant`. |
| `list` | `int64` | References `List.list`. |
| `output_position` | `int64` | 1-based within this (participant, list). Strictly increasing. |
| `word` | `string` | Uppercase. |
| `category` | `string` | The recalled word's category. Empty string for extra-list intrusions where unknown. |
| `serial_position` | `int64` | 1..W if the recall corresponds to a presented word on this list (joins to `Presented word` on participant+list+serial_position); **0** if extra-list intrusion. |
| `list_group` | `string` | Derived. |

Intrusions (`serial_position == 0`) do not contribute to the per-list log-likelihood (§4.3) but their count is recorded in the fit-result diagnostics.

### 1.5 Dataset metadata (manifest.json)

Required top-level keys in `manifest.json`:

| Key | Type | Purpose |
|-|-|-|
| `source` | object | Provenance: `paper` (citation string), `condition` (human-readable label), `egg_url` (URL), `egg_sha256` (hex). |
| `design` | object | `participants`, `lists_per_participant`, `words_per_list`, `unique_categories_per_list`, `unique_categories_total`, `early_lists` (description), `late_lists` (description). |
| `files` | object | Per-file: `{ "sha256": "<hex>", "size_bytes": <int> }` for every committed file (both Parquet and CSV). |
| `row_counts` | object | `presented`, `recalled_total`, `recalled_in_list`, `recalled_extra_list_intrusions`. |
| `created_at` | string | ISO-8601 UTC timestamp. |

A single `MANIFEST_KEYS` constant in `ms_tcm.schema` enumerates these; the validator uses it directly so the spec and the code cannot drift.

### 1.6 Model parameters

In-memory only; never serialized as part of a dataset. Held as an immutable `ModelParameters` dataclass.

| Field | Symbol | Type | Default | Constraint |
|-|-|-|-|-|
| `beta_global` | β_G | `float` | 0.5 | 0 < β_G < 1 |
| `beta_storyline` | β_S | `float` | 0.5 | 0 < β_S < 1 |
| `w_global` | w_G | `float` | 0.2 | 0 ≤ w_G ≤ 1 |
| `w_storyline` | w_S | `float` | 0.8 | abs(w_G + w_S − 1) < 1e-12 |
| `w_global_ret` | w_G^ret | `float` | inherits `w_global` | 0 ≤ w_G^ret ≤ 1 |
| `w_storyline_ret` | w_S^ret | `float` | inherits `w_storyline` | abs(w_G^ret + w_S^ret − 1) < 1e-12 |
| `gamma` | γ | `float` | 0.0 | ≥ 0; 0 ⇒ §5.1 off |
| `alpha_enabled` | — | `bool` | `False` | `True` ⇒ §5.2 on (requires a reference-edge table; free-recall default False) |
| `lambda_interference` | λ | `float` | 0.0 | ≥ 0; 0 ⇒ §5.3 off |
| `feature_dim` | d | `int` | matches dataset | > 0; matches the encoder output width (§2) |
| `seed` | — | `int64` | required | Passed to `numpy.random.Generator(PCG64(seed))`; split into two streams for (a) optimizer restarts, (b) bootstrap resamples |

The optimizer sees a **reparameterized** vector over ℝⁿ so bounds are never active (§4.4).

### 1.7 Fit result

Produced by `fit_mle` + `bootstrap_ci`; written to disk by `FitResult.save(out_dir)` as `fit_summary.json` and `fit_bootstrap.parquet`.

| Table / file | Key columns / fields | Purpose |
|-|-|-|
| `fit_summary.json` | `parameters`, `log_likelihood`, `aic`, `bic`, `n_participants`, `n_lists`, `n_recalls_used`, `n_intrusions_excluded`, `seed`, `elapsed_seconds`, `ms_tcm_version`, `dataset_manifest_sha256`, `standard_tcm` (bool) | Top-level summary |
| `parameters` sub-object | Per name: `{ "mle": <float>, "ci_lower": <float>, "ci_upper": <float>, "ci_method": "percentile_participant_bootstrap", "n_bootstraps": <int>, "n_converged": <int> }` | Per-parameter MLE + 95 % CI |
| `fit_bootstrap.parquet` | `bootstrap_id` (int64), one column per parameter name (float64), `converged` (bool) | Every bootstrap draw's estimate, for downstream analyses and alternative CI methods |

## 2. Multi-hot feature encoder (resolves U1)

`encode_features(presented_df) -> numpy.ndarray` returns one row per presented word with a fixed-width multi-hot concatenation of the word's paper-documented features. The encoder is **pure** — no hidden state — so the mapping is byte-deterministic across platforms.

The concatenation, in order, with exact sub-widths:

| Block | Width | Encoding |
|-|-|-|
| `category` | 16 | One-hot over the 15 categories in the FRFR wordpool (`BODY PARTS`, `BUILDING RELATED`, `CITIES`, `CLOTHING`, `COUNTRIES`, `FLOWERS`, `FRUITS`, `INSECTS`, `INSTRUMENTS`, `KITCHEN-RELATED`, `MAMMALS`, `STATES`, `TOOLS`, `TREES`, `VEGETABLES`) plus one `UNKNOWN` slot (reserved; 0 on the shipped data). |
| `size` | 2 | One-hot over `("small", "large")`. |
| `first_letter` | 26 | One-hot over A..Z. |
| `word_length` | 10 | One-hot over integer bins {3, 4, 5, 6, 7, 8, 9, 10, 11, 12}. Out-of-range lengths raise `ValueError`. |
| `color_r` | 4 | One-hot over uniform bins of the [0, 255] RGB space: bin i ⇔ `color_r // 64`, with the reduced-color sentinel `-1` mapped to all-zero. |
| `color_g` | 4 | Same scheme. |
| `color_b` | 4 | Same scheme. |
| `pos_x` | 2 | One-hot over `{left, right}` ⇔ `pos_x < 0.5 vs ≥ 0.5`, with NaN ⇒ all-zero. |
| `pos_y` | 2 | One-hot over `{top, bottom}` ⇔ `pos_y < 0.5 vs ≥ 0.5`, with NaN ⇒ all-zero. |

**Total**: d = 1 (list-start) + 16 + 2 + 26 + 10 + 4 + 4 + 4 + 2 + 2 = **71**. The first dimension is reserved as the list-start token `e_start` (§4.1) and is always zero in a presented word's feature vector; the remaining 70 dimensions carry the word features. This exact dimensionality is recorded in `Dataset.feature_dim` at load time and must equal `ModelParameters.feature_dim`.

The encoding choices are deliberately coarse (e.g., 4 color bins rather than 8 or 16) to keep d small (~70 rather than many hundreds) and to avoid over-fitting to visual features that are not the scientific focus of this feature. They are tunable in `ms_tcm/features.py` via module-level constants (`CATEGORY_BINS`, `LENGTH_BINS`, `COLOR_BIN_WIDTH`, `POSITION_BIN_CUT`); changing any of them is a MINOR version bump per the schema versioning policy.

## 3. Dataset entity relationships

- Each `Participant` has L `List`s. FRFR-category: L = 16.
- Each `List` has W `Presented word`s. FRFR-category: W = 16.
- Each `List` has zero or more `Recalled word`s (output order). Empty recall lists are valid.
- `Recalled word.serial_position` joins to `Presented word.(participant, list, serial_position)` when > 0; otherwise the recall is an extra-list intrusion.
- There is no "cue–target trial" entity in this feature. The free-recall paradigm generates recall sequences, not discrete cue–target pairs; §4.3 describes how the likelihood is computed from a sequence.

## 4. State machines

### 4.1 Encoding (per list)

At the start of every `(participant, list)`, the global and every storyline context are reset to a reserved **list-start unit vector** `e_start` (a fixed orthogonal basis vector reserved in the feature space so it is orthogonal to every word feature). This follows the canonical Howard & Kahana (2002) convention: TCM's drift update `ρ · c + β · c^IN` preserves unit norm only when `c` is already unit-norm, so a unit-norm initialization is required for §4.4's analytical anchor (similarity-after-one-drift = ρ) to hold from the first encoded word.

- `c_G(0) = e_start` (unit vector of length d)
- `c_S(0) = e_start` for every storyline S in this list (K = 4 storylines, one per category present)

In practice `e_start` is `numpy.eye(d)[0]` — the first basis vector — and is reserved by the feature encoder (§2) as a zero-valued block that no word feature populates, so orthogonality is exact.

Then, for each presented word `i` at `serial_position t ∈ 1..W`, with storyline identity `S* = category(i)`:

1. **Global context update** (§3.2): `c_G(t) ← ρ_G · c_G(t−1) + β_G · c^IN_i`, where ρ_G = √(1 − β_G²) and `c^IN_i = encode_features(i)` (the pre-experimental association matrix `M^TF` is the identity on the one-hot block for this feature).
2. **Active storyline context update** (§3.2): `c_{S*}(t) ← ρ_S · c_{S*}(t_prev^{S*}) + β_S · c^IN_i`, where `t_prev^{S*}` is the last time step in this list at which S* was active (or 0 if this is the first word of S* in this list; then the previous value is the zero vector).
3. **Inactive storyline contexts**: `c_S(t) = c_S(t−1)` for all S ≠ S* active in this list.
4. **Composite encoding context** (§3.3): `c_comp(i) ← w_G · c_G(t) + w_S · c_{S*}(t)`.
5. **Optional §5.1**: if γ > 0 and `t_prev^{S*} > 0` (i.e. resuming after a gap within the list), add `γ · c_{S*}(t_prev^{S*})` to the storyline update in step 2.
6. **Optional §5.2**: if `alpha_enabled` and a reference-edge table is supplied (free-recall default: not supplied), apply the reference sum.

No cross-list leakage (U5 resolved). Practice lists, if any, are excluded at the adapter layer before `encode_features` is called; the FRFR-category adapter does **not** drop any lists because the raw egg already excludes practice (the paper's first 3 practice lists are not in `exp2.egg`). This is recorded in `data/raw/frfr_category/manifest.json` via the `row_counts.presented == 7680 == 30 × 16 × 16` identity.

### 4.2 Retrieval scoring (resolves U2)

Free recall has no per-trial "cue event" and no "direction". Instead, recall proceeds as a sequence: the first recall is cued by the encoding-time context (end-of-list state); each subsequent recall is cued by the just-recalled word's encoding composite. The retrieval API mirrors this:

```python
score_first_recall(state: EncodingState, participant: int, list_: int) -> np.ndarray
score_next_recall(state: EncodingState, participant: int, list_: int,
                  last_recalled_event: int) -> np.ndarray
```

Both functions return a length-W vector of probabilities over the W presented positions plus an (implicit) "stop" mass. Specifically:

1. **Candidate set**: in this feature, the candidate set is the full list (all W presented words), not "not-yet-recalled" — this lets the likelihood handle repetitions cleanly (an observed repeat simply consumes its own probability mass). A stricter "no-repeats" variant is available as a configuration flag for ablations but is off by default.
2. **Retrieval context**: for the first recall, `c_ret = c_ret(t = W)` using the end-of-list global and active storyline contexts and the retrieval weights `w_G^ret`, `w_S^ret`. For a later recall following word `j`, `c_ret = w_G^ret · c_G(t_j) + w_S^ret · c_{S_j}(t_j)`.
3. **Unnormalized scores**: `score(i) = sim(c_ret, c_comp(i)) · a_i · exp(−λ · I_{ij})`, where `a_i` defaults to 1 and the λ factor defaults to 1 when interference is disabled. Cosine similarity with the 1e-30 epsilon guard.
4. **Normalization**: softmax with temperature 1 over the W candidates (plus optional stop mass if enabled; off by default).
5. **Empty candidate set**: in free recall the candidate set is always the full list, so an empty set cannot arise from the model. An *observed* recall sequence that exceeds W is truncated at W (the excess recalls contribute to the intrusion count, not the likelihood).

`score` returning `NaN` therefore does not arise in normal free-recall operation. It remains the documented return for pathological inputs (e.g. uninitialized context). The likelihood routine treats `NaN` as a hard error.

### 4.3 Per-list log-likelihood (resolves U4)

For one observed recall sequence `r_1, r_2, …, r_R` (excluding intrusions) from one `(participant, list)`:

```
log_L(list) = log P(r_1 | c_ret(first))           # first-recall probability
            + Σ_{k=2..R} log P(r_k | c_ret(r_{k-1}))   # subsequent recalls
```

The dataset log-likelihood is the sum of list log-likelihoods over every (participant, list). Intrusions are skipped (they do not appear in the sum and do not consume candidate probability). Empty recall lists contribute 0 to the log-likelihood.

If any per-recall probability is ≤ 0 (numerical underflow), the implementation returns `-numpy.inf` for that list, which the optimizer treats as an infeasible region; it does not crash.

### 4.4 MLE reparameterization (resolves U6)

The optimizer sees an unbounded vector. The constrained parameters are reparameterized so L-BFGS-B can run with `bounds=None`:

| Parameter | Reparameterization |
|-|-|
| β_G | `sigmoid(z_β_G)` with `z_β_G ∈ ℝ`. |
| β_S | `sigmoid(z_β_S)`. |
| w_G | `sigmoid(z_w)`; then w_S = 1 − w_G. |
| w_G^ret | `sigmoid(z_w_ret)`; then w_S^ret = 1 − w_G^ret. |
| γ | `softplus(z_γ)` ⇒ ≥ 0. |
| λ | `softplus(z_λ)` ⇒ ≥ 0. |

Convergence criterion is `scipy.optimize.OptimizeResult.success == True`. Optimizer non-convergence for a given bootstrap draw is discarded and recorded in `fit_summary.json` as `n_converged < n_bootstraps`; the overall fit aborts if fewer than 90 % of draws converge. The MLE restart starting points and the bootstrap resample indices are drawn from two independent PCG64 streams derived from the single top-level `seed` (resolves A2).

## 5. Bootstrap (participant-level)

`bootstrap_ci(dataset, *, n_bootstraps=1000, seed, ci=0.95, ...)`:

1. Enumerate participant ids.
2. For each of `n_bootstraps` draws (seeded), sample P participant ids with replacement.
3. Construct the resampled dataset view; call `fit_mle` with its own seeded restart stream.
4. Record the resulting parameter vector and `converged` flag.
5. For each parameter, return percentile CI endpoints at `ci` (default 95 %) — the `(1-ci)/2` and `1-(1-ci)/2` quantiles of the bootstrap distribution.

Non-convergent draws are excluded from the percentile computation. `ci_method` is always `"percentile_participant_bootstrap"` in v1.

## 6. Sampling synthetic recalls (resolves U3)

For parameter recovery (SC-009), `sample_recalls(model, dataset_skeleton, rng) -> pandas.DataFrame` replaces the observed `recalled.parquet` with a synthetic one generated by the model:

1. Run `encode(dataset_skeleton, parameters)`.
2. For each `(participant, list)`, sample recall length R (e.g. a Poisson draw or inherit from the skeleton).
3. Sample `r_1 ~ Categorical(score_first_recall(...))`; then iterate `r_k ~ Categorical(score_next_recall(..., r_{k-1}))`.
4. Emit a `recalled.parquet`-shaped table. `sample_recalls` is deterministic given `rng`.

## 7. Validation rules (FR-012, SC-005)

`validate_dataset(path)` MUST enforce exactly the seven rules:

1. Every required file present with the expected schema and types.
2. Every presented row has `(participant, list, serial_position)` consistent with `manifest.design` (no gaps or out-of-range values).
3. Every recalled row with `serial_position > 0` joins to an existing presented row on `(participant, list, serial_position)`.
4. `list_group` ∈ {`early`, `late`} and matches `list < 8` vs `list ≥ 8`.
5. No duplicate `(participant, list, serial_position)` in presented; no duplicate `(participant, list, output_position)` in recalled.
6. Every file's SHA-256 matches `manifest.files`.
7. `manifest.json` is well-formed JSON with the exact set of top-level keys in `MANIFEST_KEYS`.

Each violation is named by file, column, and offending row id.
