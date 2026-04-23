# Phase 1 Data Model: MS-TCM v6 Hierarchical CMR

**Feature**: 002-ms-tcm-v6-hcmr
**Date**: 2026-04-23
**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md)

This document defines the in-memory data entities the v6 implementation exposes. On-disk schemas for the FRFR-category dataset are unchanged from 001 (see [contracts/dataset-schema.md](contracts/dataset-schema.md)); this document describes only the Python-side model objects, their fields, relationships, and state transitions introduced by the v6 rewrite.

## 1. ModelParameters (v6 schema)

**File**: `code/ms_tcm/params.py`

A frozen dataclass capturing the v6 parameter inventory. Fully replaces the v1 `ModelParameters` (FR-013 destructive delete).

```python
@dataclass(frozen=True)
class ModelParameters:
    # Core hierarchical-CMR parameters (shared across paradigms)
    beta_enc: float = 0.679        # Item-level drift at encoding (C&Z 2025 Table 1)
    beta_story: float = 0.400      # Storyline-level drift at encoding (C&Z 2025 Table 1; labeled β_list)
    gamma_fc: float = 0.315        # Pre- vs experimental context mixture weight (C&Z 2025 Table 1; v6 Eq 1.5.1)
    k: float = 6.50                # Softmax inverse temperature at retrieval (C&Z 2025 Table 1)
    lambda_reinstate: float = 0.80 # Storyline-return reinstatement strength (v6 §5; NEW in MS-TCM).
                                   # [A2 convention: math symbol `λ` ↔ Python identifier `lambda_reinstate`
                                   # because `lambda` is a reserved Python keyword.]

    # Free-recall-only parameters (ignored in cued-recall paradigms)
    beta_rec: float = 0.326        # Retrieval drift rate (C&Z 2025 Table 1)
    epsilon_d: float = 1.04        # Stopping-rule rate (C&Z 2025 Table 1)

    # Structural
    feature_dim: int = 71          # Carried forward from 001 feature encoder (identity M^FC_pre)
    paradigm: Literal["free_recall", "cued_recall"] = "free_recall"
    standard_tcm: bool = False     # [A18 resolved] When True, the orchestrator skips all storyline-level
                                   # updates (drift, boundary caching, reinstatement) and treats the
                                   # entire list as a single storyline. Set by ModelParameters.standard_tcm()
                                   # classmethod. Distinct from λ=0 (which only disables reinstatement,
                                   # not storyline-level drift). FR-011.
    seed: int = 0
```

**Validation (FR-001, FR-008)**:

- `0 < beta_enc < 1` and `0 < beta_story < 1` and `beta_enc > beta_story` (enforced with explicit ValueError).
- `0 <= gamma_fc <= 1`.
- `k > 0`.
- `0 <= lambda_reinstate <= 1`.
- Free-recall path: `0 < beta_rec < 1`, `epsilon_d > 0`.
- `feature_dim > 0`.

**Reductions**:

- `ModelParameters.standard_tcm(**kwargs) -> ModelParameters` (FR-011): returns a parameter set with `lambda_reinstate=0.0`, standard-TCM marker set, which drives the hcmr orchestrator to skip storyline updates. Passing `lambda_reinstate != 0` in kwargs raises ValueError.

**Retired from v1** (FR-013 — raise AttributeError when accessed, with a docstring pointing at `notes/v6_migration.md`):

- `w_global`, `w_storyline`, `w_global_ret`, `w_storyline_ret`, the v1 `gamma` (which meant "resumption weight" in v1 §5.1 but is unrelated to v6 γ_fc), the v1 `lambda_interference` (v1 §5.3), `alpha_enabled`, `tau`, `phi_s`, `phi_d`.

## 2. MFCPreMatrix

**File**: `code/ms_tcm/preexp.py`

First-class pluggable pre-experimental matrix (FR-007, v6 §1.5). Replaces the implicit identity-only handling in v1.

```python
class MFCPreMatrix(Protocol):
    """Abstract contract for a pre-experimental item-to-context matrix."""
    n_features: int                # feature dimensionality d
    n_items: int                   # number of distinct items this matrix represents
    def apply(self, item_indices: np.ndarray) -> np.ndarray:
        """Return M^FC_pre @ f_i for each item index i, shape (len(item_indices), n_features)."""
        ...

class IdentityPreMatrix:           # default, used by the FRFR-category worked example
    """Each item's pre-experimental context is its own one-hot."""
    ...

class EmbeddingPreMatrix:          # future cued-recall path (Xu et al. 2026)
    """Pre-experimental context from pre-computed embeddings (e.g., USE)."""
    ...
```

**Mixture (v6 Eq 1.5.1)**:

```
c^IN_i = (1 - gamma_fc) * M^FC_pre @ f_i + gamma_fc * M^FC_exp @ f_i
```

`M^FC_exp` is the Hebbian-accumulated experimental matrix updated during encoding (§5 below).

## 3. Item- and storyline-level contexts

**File**: `code/ms_tcm/drift.py` (replaces `context.py`)

Two context vectors per list, both initialized to the reserved list-start unit vector `e_start` (FR-010, feature dim 0 one-hot):

```python
c_item: np.ndarray   # shape (W + 1, d); c_item[0] = e_start
c_story: np.ndarray  # shape (W + 1, d); c_story[0] = e_start
```

**Drift (v6 §2.1)**, when item i belongs to storyline s(i) and no event/storyline boundary has been crossed:

```
c_item[i] = rho_enc * c_item[i-1] + beta_enc * c^IN_i,  rho_enc = sqrt(1 - beta_enc^2)
c_story[i] = rho_story * c_story[i-1] + beta_story * c^IN_i,  rho_story = sqrt(1 - beta_story^2)
```

Inactive storylines: contexts are frozen (carried forward unchanged).

## 4. Boundary-driven state transitions

**File**: `code/ms_tcm/boundaries.py`

Three boundary mechanisms, per v6 §2.2–2.4:

### 4.1 Event boundary within a storyline (FR-002)

When `e(i) != e(i-1)` and `s(i) == s(i-1)`:

```
c_item[i] <- c_story[i]    # synchronize item to storyline
```

No separate event-level associative matrix (v6: "within-storyline event boundaries do not have their own associative matrix").

### 4.2 Storyline switch (FR-003)

When `s(i) != s(i-1)`:

1. Cache outgoing storyline context to M^SC:
   ```
   M^SC += g_{s(i-1)} @ c_story_out.T
   ```
   where `g_{s(i-1)}` is a one-hot vector identifying the outgoing storyline in the storyline-index space.
2. Synchronize item to new storyline context:
   ```
   c_item[i] <- c_story[i]
   ```

### 4.3 Storyline return — the new v6 mechanism (FR-004)

When `s(i) == s_returning` and `s_returning` has appeared before:

1. Read cached storyline context from M^SC:
   ```
   c_story_cached = (M^SC.T @ g_{s(i)})   # the cached "cached outgoing" vector
   ```
2. Reinstate with blend weight λ:
   ```
   c_story[i] <- lambda_reinstate * c_story_cached + (1 - lambda_reinstate) * c_story[i-1]
   ```
3. Synchronize item to new storyline context (same as 4.2 step 2):
   ```
   c_item[i] <- c_story[i]
   ```

## 5. Associative matrices

**File**: `code/ms_tcm/matrices.py`

### 5.1 M^IC (item → item-context)

Shape `(d_items, d)`. Updated at every encoding step (v6 Eq 3):

```
M^IC += c_item[i].reshape(-1, 1) @ f_i.reshape(1, -1)
```

Drives retrieval activation: `a = M^IC.T @ c_item_cue` (FR-005).

### 5.2 M^SC (storyline → storyline-context)

Shape `(K_storylines, d)`. Updated only at storyline switches (v6 Eq 5). See §4.2 above.

### 5.3 M^FC_exp (experimental item → pre-experimental context)

Hebbian accumulation used in the mixture (v6 §1.5, Eq 1.5.1). Shape `(d, d_items)`. Updated during encoding:

```
M^FC_exp += c^IN_i.reshape(-1, 1) @ f_i.reshape(1, -1)
```

## 6. HierarchicalCMRModel (orchestrator)

**File**: `code/ms_tcm/hcmr.py` (replaces `model.py`)

```python
class HierarchicalCMRModel:
    parameters: ModelParameters
    pre_matrix: MFCPreMatrix

    def encode(self, dataset: Dataset) -> EncodingState:
        """Run encoding across every (participant, list); return all contexts + matrices."""
    def score_first_recall(self, state: EncodingState, participant: int, list_: int) -> np.ndarray:
        """Free recall: P(j | end-of-list cue) via softmax(k * M^IC.T @ c_item_end)."""
    def score_next_recall(self, state: EncodingState, participant: int, list_: int,
                          last_recalled_serial_position: int,
                          recalled_sps: set[int] | None = None) -> np.ndarray:
        """Free recall: P(j | last recall) with beta_rec drift of retrieval context toward recalled item's c_item."""
    def score_cue(self, state: EncodingState, participant: int, list_: int, cue_serial_position: int) -> np.ndarray:
        """Cued recall: P(j | cue) with c_item_cue as the retrieval context (no beta_rec drift, no stopping rule)."""
    def sample_recalls(self, dataset_skeleton: Dataset, rng: np.random.Generator, *,
                       recall_length_fn: Callable[[int], int] | None = None,
                       stopping_epsilon_d: float | None = None) -> pa.Table:
        """FR-012: deterministically sample recalled.parquet-shaped output."""
```

`EncodingState` holds per-list trajectories (`c_item`, `c_story`), final `M^IC`, cached `M^SC`, active-storyline labels, and the per-list presented features.

## 7. Reference curves

**File**: `data/processed/reference_curves/frfr_category_{spc,pfr,lag_crp}.parquet`

Precomputed empirical free-recall curves from FRFR-category. Each file has a minimal schema:

- `spc.parquet`: columns `serial_position: int64` (1..16), `p_recall: float64`.
- `pfr.parquet`: columns `serial_position: int64` (1..16), `p_first_recall: float64`.
- `lag_crp.parquet`: columns `lag: int64` (-15..+15, excluding 0), `crp: float64`.

Companion `manifest.json` records per-file SHA-256, the source dataset's manifest hash, and the `scripts/build_reference_curves.py` invocation line.

## 8. Benchmark record

**File**: `data/processed/benchmarks/benchmark_log.csv`

Append-only CSV with one row per `scripts/benchmark_fit.py` run:

| Column | Type | Source |
|-|-|-|
| timestamp | ISO-8601 UTC string | run start |
| git_sha | string | `git rev-parse HEAD` |
| platform | `"macos"` / `"linux"` / `"windows"` | platform detection |
| tier | `"tier1"` / `"tier2-float64"` / `"tier2-float32"` | config |
| dtype | `"float64"` / `"float32"` | [A10 resolved] redundant with `tier` but written explicitly so downstream log parsers do not have to parse the tier string |
| n_participants | int | dataset |
| n_bootstraps | int | invocation |
| n_restarts | int | invocation |
| seed | int | invocation |
| wall_clock_seconds | float | measured |
| peak_memory_mb | float | measured via `resource.getrusage` |

## 9. Relationships diagram

```
             +---------+
             | Dataset |
             +---------+
                  |
                  v
        +-------------------+
        | MFCPreMatrix      |
        +-------------------+
                  |
                  v
        +-------------------+    +---------+
        | ModelParameters   |--> | fit     |--> FitResult (unchanged from 001 shape, v6 param set)
        +-------------------+    +---------+
                  |
                  v
        +---------------------+
        | HierarchicalCMRModel|
        +---------------------+
                  |
                  v
        +-----------------+
        | EncodingState   |  --> score_first_recall, score_next_recall, score_cue, sample_recalls
        | (c_item, c_story|
        |  M^IC, M^SC)    |
        +-----------------+
```
