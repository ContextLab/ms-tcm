# Contract: MLE + Bootstrap-CI Fitter (v6)

**Feature**: 002-ms-tcm-v6-hcmr
**Module**: `ms_tcm.fit`, `ms_tcm.bootstrap`

## 1. Objective

Compute the maximum-likelihood estimate of the v6 parameter vector θ = (β_enc, β_story, γ_fc, k, λ, β_rec, ε_d) for a given dataset, plus a percentile 95 % CI for each via participant-level bootstrap.

## 2. Objective function

Negative log-likelihood of the observed recall sequences given the presented sequences, summed over all (participant, list). Free-recall paradigm uses:

```
-log L(θ) = - sum_{p, ℓ} [ log P(first_recall | state(p, ℓ, θ))
                          + sum_{out_pos >= 2} log P(next_recall | last_recall, state(p, ℓ, θ))
                          + log P(stop | state, epsilon_d, θ) ]
```

Cued-recall paradigm uses `log P(recall | cue)` per trial with no β_rec or ε_d contribution.

## 3. Reparameterization (Tier 1)

The optimizer works in an unconstrained space, transformed back to the bounded parameters internally:

| Parameter | Constraint | Transform |
|-|-|-|
| β_enc | (0, 1) | logit |
| β_story | (0, β_enc) | logit on (β_story / β_enc) |
| γ_fc | [0, 1] | logit with ε clamp |
| k | > 0 | log |
| λ | [0, 1] | logit with ε clamp |
| β_rec | (0, 1) | logit (free-recall only) |
| ε_d | > 0 | log (free-recall only) |

Gradient-free L-BFGS-B (scipy) with finite-difference jacobian; Tier 2 switches this to JAX autodiff.

## 4. Restarts

`--n-restarts N` (default 5) seeded random initial points in the unconstrained space (Gaussian around the transform of the C&Z 2025 defaults). Best log-likelihood wins. Each restart gets a deterministic seed derived from `np.random.SeedSequence([seed, restart_id]).generate_state(1)[0]`.

Non-convergent restarts are flagged but do not abort the overall fit.

## 5. Bootstrap CIs

Participant-level percentile bootstrap:

1. Enumerate participants in the dataset (usually 30 for FRFR-category).
2. For each bootstrap draw b ∈ [0, n_bootstraps):
    - Sample |participants| indices with replacement via `np.random.default_rng(seed_b).choice(...)`.
    - Build a resampled Dataset from those participants (preserving their list structure).
    - Run `fit_mle` on the resampled dataset with its own restart seed stream.
    - Record per-parameter MLE in the `bootstrap_draws` table.
3. Percentile CI per parameter: `[quantile(draws, 0.025), quantile(draws, 0.975)]`.

Non-convergent draws are flagged (via a `converged: bool` column in `bootstrap_draws`); the fit aborts with `FitError` if fewer than 90 % converge.

## 6. Parallelism (Tier 1 — FR-030)

Bootstrap loop wrapped in `multiprocessing.Pool(processes=n_processes)`. Workers receive a serializable work-item dataclass composed of **numpy arrays only** (presented-row arrays, recalled-row arrays, manifest hash string, precomputed feature matrix, M^FC_pre array) — the pyarrow-backed `Dataset` object itself is NOT passed across the process boundary. Serialization uses the standard-library multiprocessing framework's default mechanism, which for numpy arrays is efficient (shared-memory when available, buffer-copy otherwise).

Each worker returns a result dataclass with the fitted θ and a `converged: bool` flag. The main process aggregates outputs in order of completion (via `Pool.imap_unordered`) but sorts by `bootstrap_id` before writing to `bootstrap_draws`. **Deterministic equivalence with the serial path is verified by a unit test** (`test_bootstrap.py::test_parallel_matches_serial_same_seed`).

## 7. Parallelism (Tier 2 — FR-032)

JAX backend replaces scipy L-BFGS-B with an Optax-based optimizer driven by `jax.grad(negative_log_likelihood)`. Bootstrap resampling is still done via `multiprocessing.Pool` (JAX is thread-unsafe for our purposes), but each worker's inner loop is JIT-compiled.

Dtype policy (Q3 resolution):

- Default: `jax.config.update("jax_enable_x64", True)` → float64 with 1e-10 cross-platform tolerance.
- Opt-in float32: environment flag gates `jax_enable_x64 = False`; tolerance loosened to 1e-8.

## 8. FitResult serialization

`FitResult.save(out_dir)` writes:

- `fit_summary.json` — sorted JSON with the v6 parameters section (see [model-api.md](model-api.md) §5), fit-level metadata, plus:
    - `backend`: `"tier1"` / `"tier2-float64"` / `"tier2-float32"`
    - `n_processes`: int used for bootstrap parallelism
- `fit_bootstrap.parquet` — one row per bootstrap draw with columns `bootstrap_id: int`, `converged: bool`, and one column per free parameter.

Both files are deterministically byte-identical given the same dataset + seed + n_bootstraps + n_restarts + backend + dtype (SC-009 reproducibility).

## 9. Required invariants

- `fit_mle(dataset, seed=S)` called twice must return identical MLE vectors (bit-exact on Tier 1; within Q3 tolerance on Tier 2).
- `bootstrap_ci(dataset, seed=S)` called twice must produce identical `bootstrap_draws` tables (bit-exact on Tier 1; within Q3 tolerance on Tier 2).
- The fit wall-clock on CI hardware must be `< 120 s` on Tier 1 for the standard invocation (FR-031 gate).
- MLE + CI for the `--standard-tcm` reduction must match a standalone CMR implementation at 1e-10 (SC-006).
- Parameter-recovery test (FR-022): when recalls are synthesized from MS-TCM with known θ_true and fitted, each component of θ_true lies inside its 95 % bootstrap CI for at least 95 % of parameters across ≥ 20 independent recovery runs.
