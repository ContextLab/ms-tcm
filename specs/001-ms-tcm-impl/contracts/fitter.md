# Contract: MLE + Bootstrap-CI Fitter

**Producer**: `ms_tcm.fit`, `ms_tcm.bootstrap`, `ms_tcm.likelihood`
**Consumer**: `ms_tcm.cli` (the `ms-tcm fit` subcommand) and any notebook or test that wants a quantitative fit.

This contract formalizes how MS-TCM is fit to a `Dataset` and how confidence intervals are reported. It is the scientific endpoint of feature 001.

## 1. Objective

Given a `Dataset` (presented + recalled) and a free model-parameter vector θ, the **objective** is the per-recall-transition negative log-likelihood:

```
NLL(θ) = − Σ_{participants, lists} list_log_likelihood(presented, recalled; θ)
```

Summed across every `(participant, list)` in the dataset. Intrusions (serial_position == 0) are skipped; empty recall lists contribute 0. Full per-list semantics are in `data-model.md` §4.3.

## 2. Free-parameter vector

The free parameters depend on the fit mode:

| Mode | Free parameters |
|-|-|
| Default MS-TCM | `beta_global`, `beta_storyline`, `w_global` (with `w_storyline = 1 − w_global`), optionally `w_global_ret` (with `w_storyline_ret = 1 − w_global_ret`). Separate retrieval weights are fit only when the caller sets `separate_retrieval_weights=True` (default False: retrieval = encoding). |
| With γ (§5.1) | Add `gamma`. |
| With λ (§5.3) | Add `lambda_interference`. |
| α (§5.2) | Not fittable from recall data alone in free recall; held at 0 unless a dataset supplies reference edges. |
| Standard TCM | `beta_global`, `w_global` only. `beta_storyline` is unused (storyline updates skipped). `w_storyline` is constrained to 0 and the associated CI is `[0.0, 0.0]`. |

## 3. Reparameterization

The optimizer sees an unbounded real vector z; θ is recovered via the transforms documented in `data-model.md` §4.4:

- β_G, β_S, w_G (and w_G^ret if fit) → `sigmoid(z)` mapping ℝ → (0, 1).
- γ, λ → `softplus(z)` mapping ℝ → (0, ∞).

This makes L-BFGS-B (`scipy.optimize.minimize`, method='L-BFGS-B') run with `bounds=None`, removing boundary-pinning edge cases. The gradient is approximated numerically by the optimizer (no analytic gradient in v1).

## 4. `fit_mle`

```python
def fit_mle(
    dataset: Dataset,
    *,
    n_restarts: int = 5,
    seed: int,
    standard_tcm: bool = False,
    optional_mechanisms: dict[str, bool] | None = None,     # {"gamma": True, "lambda": False}
    separate_retrieval_weights: bool = False,
) -> dict[str, float]: ...
```

Behavior:

1. Construct two independent `numpy.random.Generator` streams from `seed` via `np.random.SeedSequence(seed).spawn(2)` — one for **restart initialization**, one for **bootstrap resampling** (used by `bootstrap_ci`, not by `fit_mle` directly).
2. Draw `n_restarts` initial z-vectors from a moderate-width multivariate normal (σ = 2 on each logit axis; fixed in code, not user-tunable).
3. Run `scipy.optimize.minimize` from each start with `method='L-BFGS-B'`, `bounds=None`, `options={"maxiter": 500, "ftol": 1e-10}`.
4. Among the starts where `OptimizeResult.success == True`, pick the one with the lowest NLL.
5. If **no** start converged, raise `FitError` (mapped to CLI exit code 5) rather than silently returning a non-converged result.
6. Return a dict `{param_name: value}` for the free parameters; constrained parameters (e.g. `w_storyline = 1 − w_global`) are included with their derived values.

## 5. `bootstrap_ci`

```python
def bootstrap_ci(
    dataset: Dataset,
    *,
    n_bootstraps: int = 1000,
    seed: int,
    standard_tcm: bool = False,
    ci: float = 0.95,
    optional_mechanisms: dict[str, bool] | None = None,
    separate_retrieval_weights: bool = False,
) -> FitResult: ...
```

Behavior:

1. Call `fit_mle` once on the original dataset to obtain the MLE.
2. Use the bootstrap stream from step 1 of `fit_mle` (`SeedSequence(seed).spawn(2)[1]`) to draw `n_bootstraps` resampling indices, each a length-P sample-with-replacement of participant ids.
3. For each resample, construct a `Dataset` view containing only those participants (with rows duplicated when a participant is drawn more than once), then call `fit_mle` on it with its own restart stream seeded deterministically from the bootstrap index (so the i-th bootstrap's restart sequence is identical across runs).
4. Collect the per-draw parameter vector and a `converged` flag.
5. For each free parameter, compute percentile CI endpoints at `ci` (default 95 %) — the `(1-ci)/2` and `1-(1-ci)/2` quantiles, ignoring non-converged draws.
6. If fewer than 90 % of draws converge, raise `FitError` (CLI exit code 4).
7. Return a `FitResult` (see `model-api.md` §9) with the point MLE in `parameters[name].mle`, the bootstrap CI in `ci_lower` / `ci_upper`, plus `ci_method == "percentile_participant_bootstrap"`, `n_bootstraps`, `n_converged`, and aggregate statistics (log-likelihood, AIC, BIC) computed at the point MLE on the original dataset.

## 6. Fit-result file format

`FitResult.save(out_dir)` writes two files with the deterministic conventions from `dataset-schema.md` §2 and the JSON convention from `io.write_json_sorted`:

- `fit_summary.json` — every field from `contracts/model-api.md` §9 `FitResult` serialized as sorted JSON with two-space indent and trailing newline.
- `fit_bootstrap.parquet` — columns `bootstrap_id` (int64), one `float64` column per parameter name, `converged` (bool). Sorted by `bootstrap_id`. `zstd` level 1, no dictionary, no statistics.

Both files include `ms_tcm_version` and `dataset_manifest_sha256` for traceability.

## 7. AIC / BIC

- `aic = 2k − 2 log L` where `k` is the number of free parameters (depends on mode + optional mechanisms) and `log L` is the dataset log-likelihood at the MLE.
- `bic = k log N − 2 log L` where `N = n_recalls_used` (sum of non-intrusion recalls across every `(participant, list)`).

Both are written to `fit_summary.json`. Lower is better.

## 8. Determinism guarantees

- `fit_mle(dataset, seed=N)` is deterministic given the dataset bytes and `N`.
- `bootstrap_ci(dataset, seed=N, n_bootstraps=M)` is deterministic given the dataset bytes, `N`, and `M`.
- Re-running the CLI with the same arguments and a pinned library version produces byte-identical `fit_summary.json` and `fit_bootstrap.parquet`.

## 9. Parameter-recovery contract (SC-009)

Used by `test_parameter_recovery`:

1. Choose a known θ_true.
2. Run `model = MSTCMModel(ModelParameters(**θ_true))`, then `sample_recalls(model, dataset_skeleton, rng=…)` to produce a synthetic `recalled.parquet`.
3. Construct a synthetic `Dataset` from the skeleton + synthetic recalls.
4. Run `bootstrap_ci(synthetic_dataset, ...)`.
5. Assert that, for at least 95 % of free parameters, `θ_true[name]` lies in `[ci_lower, ci_upper]` (nominal coverage for a 95 % CI). The test uses a reduced `n_bootstraps` (e.g. 200) to keep CI wall-time tractable; the full-scale 1000-bootstrap recovery is a manual validation, not a CI test.
