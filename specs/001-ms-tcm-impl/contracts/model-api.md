# Contract: Model API

**Consumer**: notebooks under `code/notebooks/`, the CLI in `code/ms_tcm/cli.py`, and test code under `code/tests/`.
**Producer**: `code/ms_tcm/`.
**Language**: Python 3.11.

The public surface below is the *only* API that downstream code MAY rely on. Everything else in the package is private implementation and subject to change without a version bump.

## 1. Importable names

```python
from ms_tcm import (
    # Types
    MSTCMModel,             # ms_tcm.model — orchestrator
    ModelParameters,        # ms_tcm.params — immutable parameter dataclass
    Dataset,                # ms_tcm.dataset — typed wrapper over presented/recalled/manifest
    EncodingState,          # ms_tcm.model — per-event context-vector trajectories
    FitResult,              # ms_tcm.fit — MLE + bootstrap-CI summary

    # Dataset I/O
    load_dataset,           # ms_tcm.dataset
    save_dataset,           # ms_tcm.dataset
    load_frfr_category,     # ms_tcm.frfr — convenience loader for the bundled dataset
    validate_dataset,       # ms_tcm.schema

    # Feature encoding
    encode_features,        # ms_tcm.features

    # Similarity / readout
    cosine_similarity,      # ms_tcm.similarity
    recall_probabilities,   # ms_tcm.similarity — softmax (temperature 1)

    # Likelihood
    list_log_likelihood,    # ms_tcm.likelihood
    dataset_log_likelihood, # ms_tcm.likelihood

    # Fitting
    fit_mle,                # ms_tcm.fit — L-BFGS-B with random restarts
    bootstrap_ci,           # ms_tcm.bootstrap — participant-level bootstrap

    # Sampling (used by parameter-recovery tests)
    sample_recalls,         # ms_tcm.model — deterministic synthetic recall sampler
)
```

Each name is defined in **exactly one** module (Constitution II). `generate_dataset` and the synthetic-generator machinery are intentionally not exported; they are tracked in GitHub issue #1.

## 2. `ModelParameters`

```python
@dataclass(frozen=True)
class ModelParameters:
    beta_global:         float = 0.5
    beta_storyline:      float = 0.5
    w_global:            float = 0.2
    w_storyline:         float = 0.8
    w_global_ret:        float | None = None    # defaults to w_global  at __post_init__
    w_storyline_ret:     float | None = None    # defaults to w_storyline at __post_init__
    gamma:               float = 0.0
    alpha_enabled:       bool  = False
    lambda_interference: float = 0.0
    feature_dim:         int   = 71             # default matches encoder width (data-model.md §2)
    seed:                int   = 0

    @classmethod
    def standard_tcm(cls, **kwargs) -> "ModelParameters": ...
    # Sets w_storyline = 0.0 (and w_storyline_ret = 0.0) regardless of kwargs; raises ValueError
    # if the caller passes conflicting weights.
```

Validation raises `ValueError`:

- `0 < beta_global < 1` and `0 < beta_storyline < 1`
- `abs(w_global + w_storyline − 1) < 1e-12`
- `abs(w_global_ret + w_storyline_ret − 1) < 1e-12`
- `feature_dim > 0`
- `gamma ≥ 0`, `lambda_interference ≥ 0`

Never silently clamp or renormalize; always raise (FR-002, FR-003).

## 3. `Dataset`

```python
@dataclass(frozen=True)
class Dataset:
    presented: pa.Table            # see contracts/dataset-schema.md §3
    recalled:  pa.Table
    manifest:  dict[str, Any]      # parsed manifest.json; required keys per data-model.md §1.5

    @property
    def num_participants(self) -> int: ...
    @property
    def num_lists_per_participant(self) -> int: ...
    @property
    def num_words_per_list(self) -> int: ...
    @property
    def feature_dim(self) -> int: ...       # == 71 for datasets using the default encoder

    def categories_in_list(self, participant: int, list_: int) -> list[str]: ...
    def iter_lists(self) -> Iterator[tuple[int, int, pa.Table, pa.Table]]: ...
    # Yields (participant, list, presented_subtable, recalled_subtable) for each list.
```

Construction is typically via `load_dataset(path)` or `load_frfr_category()`. Direct construction validates the schema at `__post_init__`.

## 4. `MSTCMModel`

```python
class MSTCMModel:
    def __init__(self, parameters: ModelParameters): ...

    def encode(self, dataset: Dataset) -> EncodingState:
        """Run the encoding loop per data-model.md §4.1 and return every list's context-vector trajectory."""

    def score_first_recall(self, state: EncodingState, participant: int, list_: int) -> np.ndarray:
        """Return a length-W probability vector over the W presented positions for the first recall."""

    def score_next_recall(self, state: EncodingState, participant: int, list_: int,
                          last_recalled_serial_position: int) -> np.ndarray:
        """Return the same-shape probability vector for a recall following `last_recalled_serial_position`."""
```

Return types:

```python
@dataclass(frozen=True)
class EncodingState:
    c_global:        dict[tuple[int, int], np.ndarray]   # (participant, list) → (W+1, d)
    c_storyline:     dict[tuple[int, int], dict[str, np.ndarray]]
                                                          # (participant, list) → { category → (W+1, d) }
    c_composite:     dict[tuple[int, int], np.ndarray]   # (participant, list) → (W, d)
    active_storyline: dict[tuple[int, int], np.ndarray]  # (participant, list) → (W,) str category per step
```

`MSTCMModel` does not own a `run(dataset)` convenience method in v1; callers who want per-list recall probabilities use `dataset_log_likelihood` (for fitting) or `sample_recalls` (for synthetic data).

## 5. Dataset I/O

```python
def load_dataset(path: str | os.PathLike) -> Dataset: ...
def save_dataset(dataset: Dataset, path: str | os.PathLike) -> None: ...
def load_frfr_category(path: str | os.PathLike = "data/raw/frfr_category") -> Dataset: ...
def validate_dataset(path: str | os.PathLike) -> ValidationReport: ...
```

Behavior:

- `load_dataset` verifies every file's SHA-256 against `manifest.json`; raises on mismatch.
- `save_dataset` writes the four data files with the deterministic Parquet settings from `contracts/dataset-schema.md` §2, computes fresh SHA-256 hashes, and writes `manifest.json` last.
- `load_frfr_category` is a thin wrapper around `load_dataset` that also checks the dataset's shape matches Manning et al. 2023 expectations (30 participants, 16×16, 4 categories/list); raises if the committed data has been tampered with.
- `validate_dataset` returns `ValidationReport(ok: bool, violations: list[Violation])` listing every failed rule by file/column/row.

## 6. Feature encoding

```python
def encode_features(presented: pa.Table | pd.DataFrame) -> np.ndarray: ...
```

Returns a `(n_rows, 71)` float64 array. Column 0 is reserved for the list-start unit vector (always zero in a presented word's features); columns 1..70 carry the word features per the encoder layout. Deterministic, pure, no hidden state. Encoding layout is defined in `data-model.md` §2. The constants (`CATEGORY_BINS`, `LENGTH_BINS`, `COLOR_BIN_WIDTH`, `POSITION_BIN_CUT`) live in `ms_tcm/features.py`.

## 7. Similarity helpers

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-30) -> float: ...
def recall_probabilities(scores: np.ndarray) -> np.ndarray: ...   # softmax; temperature 1
```

Contracts: `cosine_similarity` returns 0.0 when either input has zero norm (no NaNs). `recall_probabilities` returns an array that sums to 1.0 within 1e-12 and has no NaNs unless the input was all `NaN` (in which case the caller has a numerical error upstream and the likelihood routine raises).

## 8. Likelihood

```python
def list_log_likelihood(
    presented_list: pa.Table,
    recalled_list:  pa.Table,
    parameters:     ModelParameters,
    encoding_state: EncodingState,
    participant:    int,
    list_:          int,
) -> float: ...

def dataset_log_likelihood(
    dataset:    Dataset,
    parameters: ModelParameters,
) -> float: ...
```

Per-list semantics are defined in `data-model.md` §4.3. Extra-list intrusions are skipped. Empty recall lists contribute 0. Negative-infinity values arising from numerical underflow are returned (and treated as infeasible by the optimizer), not raised.

## 9. Fitting

```python
def fit_mle(
    dataset: Dataset,
    *,
    n_restarts: int = 5,
    seed: int,
    standard_tcm: bool = False,
    optional_mechanisms: dict[str, bool] | None = None,   # e.g., {"gamma": True, "lambda": False}
) -> FitResult: ...

def bootstrap_ci(
    dataset: Dataset,
    *,
    n_bootstraps: int = 1000,
    seed: int,
    standard_tcm: bool = False,
    ci: float = 0.95,
    optional_mechanisms: dict[str, bool] | None = None,
) -> FitResult: ...
```

`FitResult` wraps both the point estimate and the bootstrap CIs:

```python
@dataclass(frozen=True)
class FitResult:
    parameters:       dict[str, dict[str, float]]   # name → {mle, ci_lower, ci_upper, ci_method, n_bootstraps, n_converged}
    log_likelihood:   float
    aic:              float
    bic:              float
    n_participants:   int
    n_lists:          int
    n_recalls_used:   int
    n_intrusions_excluded: int
    seed:             int
    elapsed_seconds:  float
    ms_tcm_version:   str
    dataset_manifest_sha256: str
    standard_tcm:     bool
    bootstrap_draws:  pa.Table     # one row per draw: bootstrap_id, <param-name>*, converged

    def save(self, out_dir: str | os.PathLike) -> None: ...
```

`save` writes `fit_summary.json` (sorted keys, two-space indent) and `fit_bootstrap.parquet`. The `standard_tcm` flag in the CLI threads through to both `fit_mle` and `bootstrap_ci` so the baseline is exercised from the same CLI.

## 10. Sampling (parameter recovery)

```python
def sample_recalls(
    model:             MSTCMModel,
    dataset_skeleton:  Dataset,
    rng:               np.random.Generator,
    *,
    recall_length_fn:  Callable[[int], int] | None = None,   # defaults to keeping the skeleton's recall counts
) -> pa.Table: ...
```

Returns a `recalled.parquet`-shaped table. Fully determined by `rng`. Used by `test_parameter_recovery`.

## 11. Stability guarantees

- Function signatures in this document are the stable surface; removal of any name is a MAJOR version bump of the package. Addition of keyword-only arguments with defaults is MINOR.
- Default values pinned here (β_G = 0.5, β_S = 0.5, w_G = 0.2, w_S = 0.8, `feature_dim = 70`, `n_restarts = 5`, `n_bootstraps = 1000`, `ci = 0.95`) match the spec's Clarifications and FRs; changing a default is a MINOR bump.
- `ci_method` is always the string `"percentile_participant_bootstrap"` in v1.
