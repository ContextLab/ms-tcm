"""Unit tests for the bootstrap CI routine (FR-018)."""

from __future__ import annotations

import numpy as np

from ms_tcm.bootstrap import bootstrap_ci

from tests.test_dataset import _tiny_dataset


def test_bootstrap_ci_endpoints_ordered() -> None:
    """ci_lower <= ci_upper for every parameter.

    Note: on pathologically small datasets (n_participants=1, n_bootstraps=3)
    the point MLE can fall outside the bootstrap CI because each resample
    converges to a different local minimum. We check endpoint ordering here,
    not MLE containment; the 95% coverage property is asserted by the
    parameter-recovery test (SC-009) on larger synthetic data.
    """
    ds = _tiny_dataset()
    result = bootstrap_ci(ds, n_bootstraps=3, n_restarts=1, seed=0)
    for name, info in result.parameters.items():
        if info["ci_method"] == "derived":
            continue
        if np.isnan(info["ci_lower"]) or np.isnan(info["ci_upper"]):
            continue
        assert info["ci_lower"] <= info["ci_upper"] + 1e-9, (
            f"ci_lower > ci_upper for {name!r}: "
            f"{info['ci_lower']} > {info['ci_upper']}"
        )


def test_bootstrap_is_deterministic_same_seed() -> None:
    """FR-018: same seed + same dataset => same MLEs + same bootstrap draws."""
    ds = _tiny_dataset()
    r1 = bootstrap_ci(ds, n_bootstraps=3, n_restarts=1, seed=123)
    r2 = bootstrap_ci(ds, n_bootstraps=3, n_restarts=1, seed=123)
    for name in r1.parameters:
        a = r1.parameters[name]["mle"]
        b = r2.parameters[name]["mle"]
        assert abs(a - b) < 1e-8, f"MLE drift for {name!r}: {a} vs {b}"


def test_standard_tcm_has_zero_storyline_weight() -> None:
    """--standard-tcm output: w_storyline.mle == 0.0 and CI == [0, 0]."""
    ds = _tiny_dataset()
    result = bootstrap_ci(ds, n_bootstraps=2, n_restarts=1, seed=0, standard_tcm=True)
    ws = result.parameters["w_storyline"]
    assert ws["mle"] == 0.0
    assert ws["ci_lower"] == 0.0
    assert ws["ci_upper"] == 0.0
