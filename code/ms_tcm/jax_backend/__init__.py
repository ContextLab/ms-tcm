"""Optional Tier-2 JAX backend for v6 hierarchical CMR (FR-032 / research §R3).

This sub-package is only importable when the ``[jax]`` extras group is
installed (``pip install -e .[jax]``). The core Tier-1 path in
``ms_tcm.hcmr`` / ``ms_tcm.fit`` is unaffected by whether JAX is available.

Entry points:

- ``encode_list_jax(beta_enc, beta_story, lambda_reinstate, gamma_fc,
   cat_indices, standard_tcm) -> (c_item_traj, m_ic)`` — JIT-compiled
  per-list encode, produces the same (c_item, M^IC) that the Tier-1
  scalar encode does (modulo float precision; 1e-10 in float64, 1e-8 in
  float32 per FR-023 / Q3).
- ``list_log_likelihood_jax(...)`` — JIT-compiled likelihood for one
  (participant, list).
- ``dataset_log_likelihood_jax(...)`` — sum over lists; uses
  ``jax.lax.scan`` or a Python loop over variable-d lists.
- ``fit_mle_jax(...)`` — Optax-L-BFGS-driven MLE using
  ``jax.grad(dataset_log_likelihood_jax)`` for analytic gradients.

Selection: ``MS_TCM_BACKEND=jax`` env var or ``--backend jax`` CLI flag
(contracts/cli.md §Global flags). Dtype: ``MS_TCM_JAX_DTYPE=float64``
(default, 1e-10 tolerance) or ``float32`` (opt-in, 1e-8 tolerance).

Numerical notes:

- Per-list d = W+1 varies across lists (W is the list length). This
  precludes a trivial ``vmap`` across lists of different sizes. We
  instead ``jit`` the per-list functions (keyed on W for cache lookup)
  and iterate lists in Python. Under FRFR-category all lists have
  W=16, so effectively one compilation.
- The auto-fallback (A6 resolution in spec.md Assumptions): if any
  import from this sub-package fails at runtime, the CLI logs a
  warning and falls back to Tier 1 — callers must handle
  ``ImportError`` / ``ModuleNotFoundError``.
"""

from __future__ import annotations

# Intentionally minimal — concrete symbols are re-exported lazily below
# to avoid importing jax at package-import time. Callers do
# ``from ms_tcm.jax_backend import fit_mle_jax`` only after confirming
# ``MS_TCM_BACKEND=jax`` and a successful ``import jax``.

__all__ = [
    "encode_list_jax",
    "list_log_likelihood_jax",
    "dataset_log_likelihood_jax",
    "fit_mle_jax",
]


def __getattr__(name):  # pragma: no cover — import-time shim only
    if name in ("encode_list_jax", "list_log_likelihood_jax",
                "dataset_log_likelihood_jax"):
        from ms_tcm.jax_backend.hcmr_jax import (
            dataset_log_likelihood_jax,
            encode_list_jax,
            list_log_likelihood_jax,
        )
        return {
            "encode_list_jax": encode_list_jax,
            "list_log_likelihood_jax": list_log_likelihood_jax,
            "dataset_log_likelihood_jax": dataset_log_likelihood_jax,
        }[name]
    if name == "fit_mle_jax":
        from ms_tcm.jax_backend.fit_jax import fit_mle_jax
        return fit_mle_jax
    raise AttributeError(f"module 'ms_tcm.jax_backend' has no attribute {name!r}")
