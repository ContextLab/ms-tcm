"""ModelParameters (v6) — the v6 parameter inventory.

Canonical source: ``notes/two_level_cmr_v6.pdf`` §5 (parameter inventory) and
``notes/CornZhan25.pdf`` Table 1 (inherited CMR starting values).

Spec: ``specs/002-ms-tcm-v6-hcmr/spec.md`` FR-001, FR-006, FR-008, FR-009, FR-011.
Data model: ``specs/002-ms-tcm-v6-hcmr/data-model.md`` §1.

Parameter inventory (symbol / identifier / source):

- β_enc  / ``beta_enc``          / C&Z 2025 Table 1 (= 0.679)
- β_story / ``beta_story``       / C&Z 2025 Table 1 labeled β_list (= 0.400)
- γ_fc   / ``gamma_fc``          / C&Z 2025 Table 1 (= 0.315)
- k      / ``k``                 / C&Z 2025 Table 1 (= 6.50)
- λ      / ``lambda_reinstate``  / v6 §5 (= 0.80) — NEW in MS-TCM
- β_rec  / ``beta_rec``          / C&Z 2025 Table 1 (= 0.326, free-recall only)
- ε_d    / ``epsilon_d``         / C&Z 2025 Table 1 (= 1.04, free-recall only)

Symbol-vs-identifier convention: math symbols use Unicode (λ, γ_fc, ε_d),
Python fields substitute valid identifiers (``lambda_reinstate`` because
``lambda`` is a reserved keyword; ``gamma_fc``, ``epsilon_d`` for style).
See ``notes/v6_migration.md`` for the full v1→v6 mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal


@dataclass(frozen=True)
class ModelParameters:
    """Immutable container for MS-TCM v6 model parameters.

    Defaults inherit from Cornell & Zhang 2025 Table 1; λ initializes from
    notes/two_level_cmr_v6.pdf §5. Every field's docstring MUST cite its
    source (FR-009 / Constitution I).
    """

    # --- Core hierarchical-CMR parameters (shared across paradigms) ---

    beta_enc: float = 0.679
    """Item-level context drift rate at encoding.

    Source: Cornell & Zhang 2025 Table 1 (β_enc = 0.679). v6 §2.1 Eq 1.

    In MS-TCM v2 this governs the PER-STORYLINE item-level context drift
    (each storyline's c^item_s drifts at this rate when items from that
    storyline are encoded). The companion ``beta_enc_global`` parameter
    controls the global cross-storyline context drift.
    """

    beta_enc_global: float = 0.400
    """Global (cross-storyline) item-level context drift rate at encoding.

    NEW in MS-TCM Iteration 2. Governs c^global drift, which advances on
    EVERY encoding step regardless of storyline (continuous time across
    the whole list). The global level treats storylines as "items" and is
    used at retrieval to select which storyline to recall from.

    Defaults to 0.400 (the C&Z β_list rate) reflecting that the global
    cross-storyline context evolves more slowly than within-storyline
    item context. Constraint at fit time: β_enc^G ∈ (0, 1); we don't
    require β_enc^G < β_enc since they govern different scales.

    In Iteration 1 MS-TCM (and C&Z baseline), this parameter is unused.
    """

    beta_story: float = 0.400
    """List-/storyline-level context drift rate at encoding.

    Source: Cornell & Zhang 2025 Table 1 (β_list = 0.400). Stored under the
    ``beta_story`` name for historical continuity with the v6 MS-TCM spec;
    the ``beta_list`` property below exposes the same value under C&Z's
    canonical name. v6 §2.1 Eq 2 / C&Z 2025 Eq 8.
    """

    beta_rein: float = 0.300
    """List-level context reactivation drift rate at retrieval.

    Source: Cornell & Zhang 2025 Table 1 (β_rein = 0.300; C&Z 2025 Eq 14).
    Used when the model reinstates a list-level context during recall —
    in final free recall only. For single-list free recall the reinstated
    context is set directly to the beginning-of-list context (e_start)
    without drift, so β_rein is unused in that mode. Included in the
    inventory for parity with C&Z's full Table 1.
    """

    gamma_fc: float = 0.315
    """Pre- vs experimental context mixture weight.

    Source: Cornell & Zhang 2025 Table 1 (γ_fc = 0.315). v6 §1.5 Eq 1.5.1.
    Controls c^IN_i = (1 − γ_fc) · M^FC_pre · f_i + γ_fc · M^FC_exp · f_i.
    """

    k: float = 6.50
    """Softmax inverse temperature (gain) at retrieval.

    Source: Cornell & Zhang 2025 Table 1 (k = 6.50). v6 §3.1 Eq 9.
    """

    lambda_reinstate: float = 0.80
    """Storyline-return reinstatement strength — NEW in MS-TCM.

    Source: notes/two_level_cmr_v6.pdf §5 (λ = 0.80 initialization). v6 §2.4
    Eq 6. When a storyline resumes after interruption, the storyline
    context is blended with its cached pre-departure value:
    c^story_s ← λ · c̃^story_s + (1−λ) · c^story_s_prev. Disabled (=0)
    under the C&Z baseline.

    Python identifier ``lambda_reinstate`` because ``lambda`` is a reserved
    Python keyword (see notes/v6_migration.md for the symbol-vs-identifier
    convention).
    """

    tau_init: float = 0.0
    """Storyline-initiation mixture probability — NEW in MS-TCM (this work).

    At recall onset, the model selects the storyline-initiation route
    with probability τ and the standard recency-initiation route with
    probability 1-τ. Under storyline-init, a storyline ŝ is sampled
    via softmax(k · M^SC · c^list_end) and the first-recall cue is
    set to c^story_ŝ instead of c^item_end.

    τ = 0 reduces MS-TCM to C&Z 2025 (single recency-initiation cue).
    Disabled by default; enable for the multi-storyline / categorical
    MS-TCM variant.
    """

    phi_s: float = 1.0
    """Primacy gradient scale (Polyn et al. 2009 / standard CMR).

    Used ONLY in the standard CMR baseline (`_likelihood_core_cmr.py`).
    Defines the per-position primacy multiplier φ_l = φ_s · exp(−φ_d · (l−1)) + 1
    that scales M^CF_exp's columns at retrieval (Polyn et al. 2009 Eq 5),
    boosting activations of items at early serial positions to produce
    primacy in the SPC.

    NOT used in C&Z 2025 or MS-TCM — those models replace the primacy
    gradient with hierarchical retrieval / storyline-init mechanisms.
    Preserved at default (1.0, no boost above baseline) when those
    models are active.
    """

    phi_d: float = 1.0
    """Primacy gradient decay rate (Polyn et al. 2009 / standard CMR).

    Companion to φ_s. Larger φ_d → faster decay of the primacy boost
    across serial positions. Polyn et al. 2009 Table 1 fit value: 0.97.
    """

    w_global: float = 0.0
    """Cross-storyline mixing weight at retrieval — NEW in MS-TCM (this work).

    Within the storyline-initiation route (route β), the within-storyline
    retrieval phase uses a mixture of the per-storyline associative
    matrix M^CF_s and the global matrix M^CF_G:

        M^CF_effective(s) = (1 − w_global) · M^CF_s + w_global · M^CF_G

    The same mixing applies to M^FC at the c^IN_rec drift target. This
    relaxes the strict per-storyline separation: at w_global = 0 the
    within-storyline phase only sees within-storyline items (strict
    hierarchy); at w_global = 1 the within-storyline phase uses only
    the global matrix (effectively allowing any item to be activated,
    just with a re-cued initial c_ret).

    Empirical motivation: FRFR-category lag-CRP shows participants make
    BOTH within-storyline and across-storyline transitions during
    within-recall flow (57% / 38% within-cat in early/late lists, NOT
    100%). The strict architecture (w_global = 0) cannot produce
    cross-storyline transitions during a within-storyline visit, so
    the model pays log-likelihood cost for every observed cross-cat
    transition via the more-restrictive storyline-selection mechanism.
    Allowing w_global > 0 lets the model fit the observed mixing rate.

    Disabled by default (w_global = 0) to preserve the strict-hierarchy
    baseline behavior.
    """

    # --- Free-recall-only parameters ---

    beta_rec: float = 0.326
    """Within-trial retrieval context drift rate (free-recall only).

    Source: Cornell & Zhang 2025 Table 1 (β_rec = 0.326). C&Z 2025 Eq 3. In
    free recall the retrieval context drifts toward the just-recalled item's
    encoding context at rate β_rec; in cued recall this parameter is ignored.
    """

    epsilon_d: float = 1.04
    """Stopping-rule rate (free-recall only).

    Source: Cornell & Zhang 2025 Table 1 (ε_d = 1.04). C&Z 2025 Eq 7. The
    per-attempt stopping probability is p_stop = exp(−ε_d · a^nr / a^r). In
    cued recall (one response per cue) this parameter is ignored.
    """

    # --- Structural ---

    feature_dim: int = 71
    """Feature-vector dimensionality d.

    Inherited from the 001-feature encoder (15 categories + 1 UNKNOWN + 2
    sizes + 26 letters + 10 length bins + 12 color bins + 4 position bins +
    1 list-start slot = 71). Changing this is a MINOR schema-version bump
    per contracts/dataset-schema.md.
    """

    paradigm: Literal["free_recall", "cued_recall"] = "free_recall"
    """Retrieval paradigm.

    ``free_recall`` (default) — uses β_rec + ε_d stopping rule, multi-recall
    per list. ``cued_recall`` — single response per cue; β_rec and ε_d are
    ignored. Source: v6 §3 (retrieval is instruction-blind; paradigm
    differs only in whether multi-response + stopping is active).
    """

    standard_tcm: bool = False
    """When True, the orchestrator skips all storyline-level updates.

    A2/A18 resolution in specs/002-ms-tcm-v6-hcmr/data-model.md §1. Set by
    ``ModelParameters.standard_tcm()``. Distinct from λ=0 alone: λ=0 only
    disables storyline-return reinstatement, whereas this flag also bypasses
    storyline-context drift and M^SC caching entirely (treats the list as a
    single storyline). FR-011.
    """

    seed: int = 0
    """PRNG seed for deterministic restarts and bootstrap draws."""

    # --- Property aliases exposing C&Z 2025 canonical names ---

    @property
    def beta_list(self) -> float:
        """Alias for ``beta_story``, matching C&Z 2025 canonical naming.

        C&Z's Table 1 uses ``β_list``; v6 MS-TCM renamed this to ``β_story``
        to reflect narrative-storyline semantics. The underlying parameter
        is the same rate (list-level context drift at encoding); this
        property lets C&Z-aligned code (``_likelihood_core.py``) use the
        canonical name without a schema change.
        """
        return float(self.beta_story)

    # ---

    def __post_init__(self) -> None:
        # Drift-rate bounds.
        if not (0.0 < self.beta_enc < 1.0):
            raise ValueError(
                f"beta_enc must be in (0, 1); got {self.beta_enc!r}"
            )
        if not (0.0 < self.beta_enc_global < 1.0):
            raise ValueError(
                f"beta_enc_global must be in (0, 1); got "
                f"{self.beta_enc_global!r}"
            )
        if not (0.0 < self.beta_story < 1.0):
            raise ValueError(
                f"beta_story must be in (0, 1); got {self.beta_story!r}"
            )
        # v6 §2.1: β_enc > β_story (item drifts faster than storyline).
        if not (self.beta_enc > self.beta_story):
            raise ValueError(
                f"v6 §2.1 requires beta_enc > beta_story; got "
                f"beta_enc={self.beta_enc!r}, beta_story={self.beta_story!r}"
            )

        # Mixture weight bounds.
        if not (0.0 <= self.gamma_fc <= 1.0):
            raise ValueError(
                f"gamma_fc must be in [0, 1]; got {self.gamma_fc!r}"
            )
        if not (0.0 <= self.lambda_reinstate <= 1.0):
            raise ValueError(
                f"lambda_reinstate must be in [0, 1]; got {self.lambda_reinstate!r}"
            )
        if not (0.0 <= self.tau_init <= 1.0):
            raise ValueError(
                f"tau_init must be in [0, 1]; got {self.tau_init!r}"
            )
        if not (0.0 <= self.w_global <= 1.0):
            raise ValueError(
                f"w_global must be in [0, 1]; got {self.w_global!r}"
            )
        if not (0.0 <= self.beta_rein <= 1.0):
            raise ValueError(
                f"beta_rein must be in [0, 1]; got {self.beta_rein!r}"
            )

        # k > 0.
        if not (self.k > 0.0):
            raise ValueError(f"k must be > 0; got {self.k!r}")

        # Free-recall parameters (relaxed bounds for cued-recall).
        if self.paradigm == "free_recall":
            if not (0.0 < self.beta_rec < 1.0):
                raise ValueError(
                    f"beta_rec must be in (0, 1) in free-recall paradigm; "
                    f"got {self.beta_rec!r}"
                )
            if not (self.epsilon_d > 0.0):
                raise ValueError(
                    f"epsilon_d must be > 0 in free-recall paradigm; got "
                    f"{self.epsilon_d!r}"
                )

        if self.feature_dim <= 0:
            raise ValueError(
                f"feature_dim must be positive; got {self.feature_dim!r}"
            )

        if self.paradigm not in ("free_recall", "cued_recall"):
            raise ValueError(
                f"paradigm must be 'free_recall' or 'cued_recall'; "
                f"got {self.paradigm!r}"
            )

    @classmethod
    def standard_tcm_reduction(cls, **kwargs: Any) -> "ModelParameters":
        """Return a parameter set that reduces MS-TCM to standard CMR.

        Sets ``lambda_reinstate=0.0`` AND ``standard_tcm=True``. Forbids
        overriding either from ``kwargs`` to keep the reduction explicit.

        FR-011: this is the baseline for paper-level model comparison.
        """
        forbidden = {"lambda_reinstate": 0.0, "standard_tcm": True}
        for key, required in forbidden.items():
            if key in kwargs and kwargs[key] != required:
                raise ValueError(
                    f"standard_tcm_reduction() is incompatible with "
                    f"{key}={kwargs[key]!r}; this classmethod forces "
                    f"{key}={required!r}"
                )
        kwargs.update(forbidden)
        return cls(**kwargs)

    def replace(self, **changes: Any) -> "ModelParameters":
        """Return a new ModelParameters with the given fields replaced (re-validated)."""
        return replace(self, **changes)
