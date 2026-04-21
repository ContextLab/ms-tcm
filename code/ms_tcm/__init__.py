"""Multi-Stream Temporal Context Model (MS-TCM).

Public API surface — see ``specs/001-ms-tcm-impl/contracts/model-api.md``.

Names are loaded lazily so the package remains importable during
incremental development (individual submodules land in their own tasks).
Once the full implementation is in place, every name below resolves eagerly.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0.dev0"

# Map public name -> (module, attribute)
_LAZY: dict[str, tuple[str, str]] = {
    "ModelParameters":         ("ms_tcm.params", "ModelParameters"),
    "cosine_similarity":       ("ms_tcm.similarity", "cosine_similarity"),
    "recall_probabilities":    ("ms_tcm.similarity", "recall_probabilities"),
    "encode_features":         ("ms_tcm.features", "encode_features"),
    "Dataset":                 ("ms_tcm.dataset", "Dataset"),
    "load_dataset":            ("ms_tcm.dataset", "load_dataset"),
    "save_dataset":            ("ms_tcm.dataset", "save_dataset"),
    "validate_dataset":        ("ms_tcm.schema", "validate_dataset"),
    "MSTCMModel":              ("ms_tcm.model", "MSTCMModel"),
    "EncodingState":           ("ms_tcm.model", "EncodingState"),
    "sample_recalls":          ("ms_tcm.model", "sample_recalls"),
    "load_frfr_category":      ("ms_tcm.frfr", "load_frfr_category"),
    "list_log_likelihood":     ("ms_tcm.likelihood", "list_log_likelihood"),
    "dataset_log_likelihood":  ("ms_tcm.likelihood", "dataset_log_likelihood"),
    "fit_mle":                 ("ms_tcm.fit", "fit_mle"),
    "FitResult":               ("ms_tcm.fit", "FitResult"),
    "bootstrap_ci":            ("ms_tcm.bootstrap", "bootstrap_ci"),
}

__all__ = sorted(list(_LAZY.keys()) + ["__version__"])


def __getattr__(name: str):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        module = importlib.import_module(mod_name)
        value = getattr(module, attr)
        globals()[name] = value  # cache for next access
        return value
    raise AttributeError(f"module 'ms_tcm' has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover
    from ms_tcm.params import ModelParameters  # noqa: F401
    from ms_tcm.similarity import cosine_similarity, recall_probabilities  # noqa: F401
    from ms_tcm.features import encode_features  # noqa: F401
    from ms_tcm.dataset import Dataset, load_dataset, save_dataset  # noqa: F401
    from ms_tcm.schema import validate_dataset  # noqa: F401
    from ms_tcm.model import MSTCMModel, EncodingState, sample_recalls  # noqa: F401
    from ms_tcm.frfr import load_frfr_category  # noqa: F401
    from ms_tcm.likelihood import list_log_likelihood, dataset_log_likelihood  # noqa: F401
    from ms_tcm.fit import fit_mle, FitResult  # noqa: F401
    from ms_tcm.bootstrap import bootstrap_ci  # noqa: F401
