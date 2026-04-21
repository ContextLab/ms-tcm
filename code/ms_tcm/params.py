"""ModelParameters — the single source of truth for MS-TCM parameter values.

See ``specs/001-ms-tcm-impl/contracts/model-api.md`` §2 and
``specs/001-ms-tcm-impl/data-model.md`` §1.6.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

_WEIGHT_TOL = 1e-12


@dataclass(frozen=True)
class ModelParameters:
    """Immutable container for MS-TCM model parameters.

    Symbols mirror ``notes/ms-tcm.pdf`` (Constitution Principle III).
    """

    beta_global: float = 0.5
    beta_storyline: float = 0.5
    w_global: float = 0.2
    w_storyline: float = 0.8
    w_global_ret: float | None = None
    w_storyline_ret: float | None = None
    gamma: float = 0.0
    alpha_enabled: bool = False
    lambda_interference: float = 0.0
    feature_dim: int = 71
    seed: int = 0

    def __post_init__(self) -> None:
        # Inherit retrieval weights from encoding weights when not supplied.
        if self.w_global_ret is None:
            object.__setattr__(self, "w_global_ret", float(self.w_global))
        if self.w_storyline_ret is None:
            object.__setattr__(self, "w_storyline_ret", float(self.w_storyline))

        # Bounds on drift rates.
        if not (0.0 < self.beta_global < 1.0):
            raise ValueError(
                f"beta_global must be in (0, 1); got {self.beta_global!r}"
            )
        if not (0.0 < self.beta_storyline < 1.0):
            raise ValueError(
                f"beta_storyline must be in (0, 1); got {self.beta_storyline!r}"
            )

        # Mixing weight constraints (sum-to-1 within 1e-12; no silent clamp).
        if not (0.0 <= self.w_global <= 1.0):
            raise ValueError(
                f"w_global must be in [0, 1]; got {self.w_global!r}"
            )
        if not (0.0 <= self.w_storyline <= 1.0):
            raise ValueError(
                f"w_storyline must be in [0, 1]; got {self.w_storyline!r}"
            )
        if abs(self.w_global + self.w_storyline - 1.0) > _WEIGHT_TOL:
            raise ValueError(
                "encoding weights must sum to 1 within 1e-12; "
                f"got w_global={self.w_global!r}, w_storyline={self.w_storyline!r}"
            )
        if not (0.0 <= self.w_global_ret <= 1.0):
            raise ValueError(
                f"w_global_ret must be in [0, 1]; got {self.w_global_ret!r}"
            )
        if not (0.0 <= self.w_storyline_ret <= 1.0):
            raise ValueError(
                f"w_storyline_ret must be in [0, 1]; got {self.w_storyline_ret!r}"
            )
        if abs(self.w_global_ret + self.w_storyline_ret - 1.0) > _WEIGHT_TOL:
            raise ValueError(
                "retrieval weights must sum to 1 within 1e-12; "
                f"got w_global_ret={self.w_global_ret!r}, "
                f"w_storyline_ret={self.w_storyline_ret!r}"
            )

        # Feature dimensionality + optional mechanism bounds.
        if self.feature_dim <= 0:
            raise ValueError(
                f"feature_dim must be positive; got {self.feature_dim!r}"
            )
        if self.gamma < 0.0:
            raise ValueError(f"gamma must be >= 0; got {self.gamma!r}")
        if self.lambda_interference < 0.0:
            raise ValueError(
                f"lambda_interference must be >= 0; got {self.lambda_interference!r}"
            )

    @classmethod
    def standard_tcm(cls, **kwargs: Any) -> "ModelParameters":
        """Return a parameter set constrained to standard TCM (w_S = 0).

        Any ``w_global`` / ``w_storyline`` / ``w_global_ret`` / ``w_storyline_ret``
        values supplied via ``kwargs`` that conflict with the standard-TCM
        reduction raise ``ValueError`` — the reduction is explicit, not silent.
        """
        forbidden = {
            "w_global": 1.0,
            "w_storyline": 0.0,
            "w_global_ret": 1.0,
            "w_storyline_ret": 0.0,
        }
        for key, required in forbidden.items():
            if key in kwargs and kwargs[key] != required:
                raise ValueError(
                    f"standard_tcm() is incompatible with {key}={kwargs[key]!r}; "
                    f"this classmethod forces {key}={required!r}"
                )
        kwargs.update(forbidden)
        # beta_storyline is unused in standard TCM but must still satisfy 0 < β < 1;
        # inherit from the default unless the caller supplies one.
        return cls(**kwargs)

    def replace(self, **changes: Any) -> "ModelParameters":
        """Return a new ModelParameters with the given fields replaced (re-validated)."""
        return replace(self, **changes)
