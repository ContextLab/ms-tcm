# Contract: Behavioral Regression Tests (Q2 Two-Layer Policy)

**Feature**: 002-ms-tcm-v6-hcmr
**Test file**: `code/tests/test_behavioral_regression.py`
**Covers**: FR-020, FR-021, SC-001, SC-006

## 1. Scope

Verify that MS-TCM, fit to the bundled FRFR-category dataset, reproduces the three canonical free-recall phenomena: **serial position curve (SPC)**, **probability of first recall (pFR)**, and **lag-conditional response probability (lag-CRP)**. Tolerance policy is two-layer, per the Q2 clarification.

## 2. Layer 1 — Qualitative shape assertions

**Runtime target**: < 10 seconds.

Setup:
- 3-participant subset of FRFR-category.
- Fit MS-TCM with `--n-bootstraps 5 --n-restarts 2 --seed 42`.
- Simulate 500 recall sessions per list via `sample_recalls`.
- Compute SPC, pFR, lag-CRP from simulated recalls.

Assertions (W = 16 for FRFR-category):

```python
assert spc[0]  > spc[W // 2]   # detectable primacy
assert spc[-1] > spc[W // 2]   # detectable recency
assert int(np.argmax(pfr)) > W // 2                                # pFR biased to end
assert int(np.argmax(crp[crp.index != 0])) == +1                   # peak at lag +1
assert crp[+1] > crp[-1]                                           # forward asymmetry
assert crp[+1] > crp[+2]                                           # monotone falloff forward
```

Each assertion cites a canonical reference in its error message: Cornell & Zhang 2025 Figure 2a/b/c, Kahana et al. 2002 Figure 1, or the FRFR-category dataset manifest.

## 3. Layer 2 — Relative-fit against FRFR-category empirical curves

**Runtime target**: < 90 seconds.

Setup:
- Full FRFR-category dataset.
- Load empirical reference curves from `data/processed/reference_curves/frfr_category_{spc,pfr,lag_crp}.parquet` (verified against `manifest.json` SHA-256).
- Fit MS-TCM with `--n-bootstraps 50 --n-restarts 3 --seed 42` (medium fit).
- Fit standard-CMR with `--standard-tcm --n-bootstraps 50 --n-restarts 3 --seed 42`.
- Simulate recalls from each fit (same seed).
- Compute SPC, pFR, lag-CRP from each set of simulated recalls.

Assertions, for each curve ∈ {SPC, pFR, lag-CRP}:

```python
rel_err_mstcm    = np.abs(sim_mstcm    - empirical) / np.maximum(empirical, 1e-6)
rel_err_standard = np.abs(sim_standard - empirical) / np.maximum(empirical, 1e-6)

# Gate 1: MS-TCM per-bin relative error <= 20%
assert np.all(rel_err_mstcm <= 0.20), f"{curve_name} relative error exceeds 20% at bins {np.where(rel_err_mstcm > 0.20)[0]}"

# Gate 2: MS-TCM no-worse-than standard-CMR on total absolute error
assert rel_err_mstcm.sum() <= rel_err_standard.sum(), (
    f"{curve_name}: MS-TCM total error {rel_err_mstcm.sum():.3f} > standard-CMR total error {rel_err_standard.sum():.3f}"
)
```

On failure, the test prints:

```
FAIL: {curve_name} ({layer})
  per-bin relative errors: {array}
  max error at bin {idx}: MS-TCM={val}, empirical={val}
  reference: notes/CornZhan25.pdf Figure 2{a|b|c} / Kahana2002 Figure 1{a|b|c} / data/raw/frfr_category/manifest.json
```

## 4. Passes under `--standard-tcm` (FR-021)

An additional parameterization of the Layer 2 test uses the `--standard-tcm` fit as the *primary* model and skips Gate 2 (the MS-TCM no-worse-than comparison is trivially satisfied). This catches CMR-layer bugs: if standard-CMR doesn't reproduce SPC/pFR/lag-CRP, the problem is in the CMR machinery (`retrieval.py`, `matrices.py`), not the MS-TCM extension.

## 5. Run profile

| Profile | Invocation | Layer 1 | Layer 2 | Total |
|-|-|-|-|-|
| Default CI | `pytest code/tests/test_behavioral_regression.py` | ✓ | ✓ | < 3 min |
| Fast smoke | `pytest -m "not slow" code/tests/test_behavioral_regression.py::test_layer_1_shape` | ✓ | — | < 15 s |
| Full validation | `pytest code/tests/test_behavioral_regression.py --validate-full` | ✓ | ✓ + full 1000-bootstrap fit | ~10 min |

The default CI profile runs both layers.

## 6. Reference curve regeneration

Layer 2 reference curves live at `data/processed/reference_curves/`. Regenerate via:

```bash
python scripts/build_reference_curves.py data/raw/frfr_category --out data/processed/reference_curves/
```

The script is idempotent. If the empirical FRFR-category data changes (e.g. the reformat script re-runs with a corrected egg), the reference curves regenerate and the test picks up the new targets.

## 7. Failure-mode matrix

| Failure | Likely root cause | Resolution pointer |
|-|-|-|
| Layer 1 `spc[0] > spc[W//2]` false | No primacy. Missing boundary sync or wrong β_enc. | `boundaries.py`, `drift.py` |
| Layer 1 `argmax(crp)==+1` false | No forward asymmetry. Missing β_rec retrieval drift. | `retrieval.py` |
| Layer 2 SPC relative error > 20 % | Fit not converged, or feature encoder bug. | Check `fit_summary.json:log_likelihood`; rerun with more restarts. |
| Layer 2 MS-TCM worse than standard | λ regime wrong for FRFR-category. | Examine `lambda_reinstate.mle`; if near 0, storylines are not being exploited. |
| Layer 2 standard-CMR worse than MS-TCM on all curves | Inverted assertion (bug in test). Not a model issue. | — |
