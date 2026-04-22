"""Cross-platform regression fixture (T053).

This test pins the MLE that ``ms-tcm fit data/raw/frfr_category --seed 42
--n-bootstraps 40 --n-restarts 2`` produces on macOS so that the same
command on Ubuntu and Windows CI machines must agree within numerical
tolerance.

The fixture is populated from a real macOS fit; when the fit dependencies
change (feature encoder, optimizer settings, parameter set), re-run the
fit locally on macOS and update the ``EXPECTED_MLE`` dict below.

The test is marked ``slow`` because it runs a short fit
(``n_bootstraps=5, n_restarts=1``) to avoid multi-hour CI runs; the
assertion is on the point MLE, not the CI. The 5-bootstrap version is
run on every platform and the resulting per-parameter ``mle`` values
must match the fixture within 1e-6.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ms_tcm import load_frfr_category
from ms_tcm.fit import fit_mle


# Populated from a macOS run with
#     ms-tcm fit data/raw/frfr_category --seed 42 --n-bootstraps 5 \
#         --n-restarts 1 --out /tmp/ci_fixture
# executed against the corrected FRFR-category dataset (commit 405e272
# onward). The seed and bootstrap counts are fixed so CI can reproduce
# the same MLE bit-for-bit.
EXPECTED_MLE: dict[str, float] | None = {
    # macOS fit (seed=42, n_restarts=1) against commit 405e272 or later.
    # The optimizer drives w_global to its lower boundary (effectively 0) and
    # the MLE therefore corresponds to the MS-TCM regime in which the storyline
    # context carries all of the cue weight -- consistent with notes/ms-tcm.pdf
    # section 4.3 regime (a) (w_S >> w_G).
    "beta_global": 0.992682,
    "beta_storyline": 0.533453,
    "w_global": 0.000001,
    "w_storyline": 0.999999,
    "w_global_ret": 0.000001,
    "w_storyline_ret": 0.999999,
    "gamma": 0.000000,
    "lambda_interference": 0.000000,
}

EXPECTED_SEED = 42
EXPECTED_N_RESTARTS = 1


@pytest.mark.slow
@pytest.mark.skipif(
    EXPECTED_MLE is None,
    reason="Fixture not yet populated; run an MS-TCM fit on macOS and paste "
           "the MLEs into EXPECTED_MLE before enabling this test.",
)
def test_cross_platform_mle_matches_fixture() -> None:
    """The MLE produced on this machine matches the macOS fixture within 1e-6."""
    ds = load_frfr_category()
    theta = fit_mle(ds, n_restarts=EXPECTED_N_RESTARTS, seed=EXPECTED_SEED)
    for name, expected in (EXPECTED_MLE or {}).items():
        actual = float(theta[name])
        assert math.isclose(actual, expected, abs_tol=1e-6, rel_tol=0.0), (
            f"{name}: cross-platform drift {actual} vs fixture {expected}"
        )
