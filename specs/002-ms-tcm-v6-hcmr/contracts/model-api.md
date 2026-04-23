# Contract: `ms_tcm` Public Python API (v6)

**Feature**: 002-ms-tcm-v6-hcmr
**Scope**: names re-exported from `code/ms_tcm/__init__.py`. Anything not listed here is an implementation detail and may change without a semver bump.

## 1. ModelParameters

**Module**: `ms_tcm.params`

```python
from ms_tcm import ModelParameters

params = ModelParameters()                          # C&Z 2025 Table 1 defaults + v6 λ=0.80
params = ModelParameters(beta_enc=0.7, lambda_reinstate=0.5)
params = ModelParameters.standard_tcm()             # λ=0, one-storyline reduction
```

Fields: `beta_enc`, `beta_story`, `gamma_fc`, `k`, `lambda_reinstate`, `beta_rec`, `epsilon_d`, `feature_dim`, `paradigm`, `seed`. Frozen dataclass. Full schema in [data-model.md](../data-model.md) §1.

## 2. MFCPreMatrix / IdentityPreMatrix / EmbeddingPreMatrix

**Module**: `ms_tcm.preexp`

```python
from ms_tcm import IdentityPreMatrix, EmbeddingPreMatrix

pre = IdentityPreMatrix(n_items=15, n_features=71)
pre = EmbeddingPreMatrix.from_parquet("embeddings.parquet")
```

Protocol: `apply(item_indices: np.ndarray) -> np.ndarray`. Full schema in [data-model.md](../data-model.md) §2.

## 3. HierarchicalCMRModel

**Module**: `ms_tcm.hcmr`

```python
from ms_tcm import HierarchicalCMRModel, ModelParameters, IdentityPreMatrix, load_frfr_category

model = HierarchicalCMRModel(
    parameters=ModelParameters(),
    pre_matrix=IdentityPreMatrix(n_items=15, n_features=71),
)
dataset = load_frfr_category()
state = model.encode(dataset)
probs = model.score_first_recall(state, participant=0, list_=3)
```

Methods: `encode`, `score_first_recall`, `score_next_recall`, `score_cue`, `sample_recalls`. See [data-model.md](../data-model.md) §6 for signatures.

## 4. Dataset + FRFR loader (unchanged from 001)

```python
from ms_tcm import Dataset, load_dataset, load_frfr_category, save_dataset, validate_dataset
```

## 5. Likelihood + fit

**Module**: `ms_tcm.likelihood`, `ms_tcm.fit`, `ms_tcm.bootstrap`

```python
from ms_tcm import dataset_log_likelihood, fit_mle, bootstrap_ci

ll = dataset_log_likelihood(dataset, parameters)
fit_result = fit_mle(dataset, n_restarts=5, seed=42)
fit_result = bootstrap_ci(dataset, n_bootstraps=1000, seed=42)
```

`fit_mle` returns a `FitResult` with per-parameter MLE, log-likelihood, AIC, BIC (bootstrap `ci_lower` / `ci_upper` are `None`). `bootstrap_ci` returns the full `FitResult` with percentile CIs and a `bootstrap_draws` pyarrow table.

`FitResult` fields (v6 schema — same shape as v1 001 but v6 parameter names):

```python
FitResult(
    parameters={
        "beta_enc":         {"mle": ..., "ci_lower": ..., "ci_upper": ..., "ci_method": ..., "n_bootstraps": ..., "n_converged": ...},
        "beta_story":       {...},
        "gamma_fc":         {...},
        "k":                {...},
        "lambda_reinstate": {...},
        "beta_rec":         {...},  # free-recall only
        "epsilon_d":        {...},  # free-recall only
    },
    log_likelihood=...,
    aic=...,
    bic=...,
    n_participants=...,
    n_lists=...,
    n_recalls_used=...,
    n_intrusions_excluded=...,
    seed=...,
    elapsed_seconds=...,
    ms_tcm_version=...,
    dataset_manifest_sha256=...,
    standard_tcm=False,
    backend="tier1" | "tier2-float64" | "tier2-float32",
    bootstrap_draws=<pyarrow.Table>,
)
```

## 6. Benchmarking

**Module**: `ms_tcm.benchmark`

```python
from ms_tcm import benchmark_fit

record = benchmark_fit(
    dataset_path="data/raw/frfr_category",
    tier="tier1",              # or "tier2-float64" / "tier2-float32"
    n_bootstraps=1000,
    n_restarts=5,
    seed=42,
)
# record is a dict with: timestamp, git_sha, platform, tier, n_bootstraps, n_restarts,
#                       seed, wall_clock_seconds, peak_memory_mb
```

## 7. Behavioral analyses

**Module**: `ms_tcm.analyses.spc`, `ms_tcm.analyses.pfr`, `ms_tcm.analyses.lag_crp`

```python
from ms_tcm.analyses import compute_spc, compute_pfr, compute_lag_crp

# From dataset (empirical):
spc = compute_spc(dataset)                # pd.Series indexed by serial position (1..W)
pfr = compute_pfr(dataset)
lag_crp = compute_lag_crp(dataset)

# From simulated recalls:
spc_sim = compute_spc(simulated_recalls)  # same signature, different input provenance
```

## 8. Retired from v1 (FR-013)

Importing any of the following raises `ImportError` with a pointer to `notes/v6_migration.md`:

- `ms_tcm.composite` (entire module)
- `ms_tcm.model.MSTCMModel` (renamed to `HierarchicalCMRModel` in `ms_tcm.hcmr`)
- `ms_tcm.context.update_global_context`, `ms_tcm.context.update_storyline_context` (replaced by `ms_tcm.drift`)
- `ms_tcm.mechanisms.apply_resumption_reinstatement`, `ms_tcm.mechanisms.apply_conversational_references`, `ms_tcm.mechanisms.interference_factor` (subsumed into v6 λ and retrieval route)
- `ms_tcm.similarity.cosine_similarity`, `ms_tcm.similarity.recall_probabilities` (subsumed into `ms_tcm.retrieval`)

## 9. Versioning

`ms_tcm.__version__` bumps to `"0.2.0"` when this feature merges (semantic-versioning MINOR — API-breaking from 001, but pre-1.0 so MINOR is acceptable). A MAJOR bump to `1.0.0` is deferred until the Xu et al. 2026 cued-recall fit lands.
