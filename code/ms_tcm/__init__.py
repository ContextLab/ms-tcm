"""Multi-Stream Temporal Context Model (MS-TCM) — v6 hierarchical CMR.

Canonical spec: ``notes/two_level_cmr_v6.pdf``.
Theoretical predecessor: ``notes/CornZhan25.pdf`` (Cornell & Zhang 2025).
v1 migration notes: ``notes/v6_migration.md``.

Public API surface — see ``specs/002-ms-tcm-v6-hcmr/contracts/model-api.md``.

Names are loaded lazily so the package remains importable during incremental
development (v6 submodules land across Phases 2 and 3 of the 002 feature).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.2.0.dev0"

# Map public name -> (module, attribute). Only v1-neutral names during the
# Phase-1 window; v6 names (ModelParameters, HierarchicalCMRModel, etc.) are
# added as their modules land in Phase 2 and Phase 3.
_LAZY: dict[str, tuple[str, str]] = {
    "encode_features":         ("ms_tcm.features", "encode_features"),
    "Dataset":                 ("ms_tcm.dataset", "Dataset"),
    "load_dataset":            ("ms_tcm.dataset", "load_dataset"),
    "save_dataset":            ("ms_tcm.dataset", "save_dataset"),
    "validate_dataset":        ("ms_tcm.schema", "validate_dataset"),
    "load_frfr_category":      ("ms_tcm.frfr", "load_frfr_category"),
}

__all__ = sorted(list(_LAZY.keys()) + ["__version__"])


def __getattr__(name: str):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        module = importlib.import_module(mod_name)
        value = getattr(module, attr)
        globals()[name] = value  # cache for next access
        return value
    # Retired v1 symbols should raise with a pointer to the migration note.
    _RETIRED_V1 = {
        "MSTCMModel", "EncodingState", "sample_recalls",
        "ModelParameters", "cosine_similarity", "recall_probabilities",
        "list_log_likelihood", "dataset_log_likelihood",
        "fit_mle", "FitResult", "bootstrap_ci",
    }
    if name in _RETIRED_V1:
        raise AttributeError(
            f"{name!r} was retired or moved during the v6 rewrite "
            f"(feature 002-ms-tcm-v6-hcmr). See notes/v6_migration.md for the "
            f"v1-to-v6 symbol mapping. The v6 replacement will land in Phase 2/3."
        )
    raise AttributeError(f"module 'ms_tcm' has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover
    from ms_tcm.features import encode_features  # noqa: F401
    from ms_tcm.dataset import Dataset, load_dataset, save_dataset  # noqa: F401
    from ms_tcm.schema import validate_dataset  # noqa: F401
    from ms_tcm.frfr import load_frfr_category  # noqa: F401
