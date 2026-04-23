# Specification Quality Checklist: MS-TCM v6 (Hierarchical CMR) Rewrite + Fast Inference + Paper/Docs Sync

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-04-22
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [ ] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

**Notes on "no implementation details"**: This feature is explicitly a cross-cutting engineering/scientific rewrite. The spec mentions Python, JAX, multiprocessing.Pool, `scripts/`, `code/ms_tcm/`, and file paths because the feature's *primary deliverables* are source-code artifacts in a scientific Python library plus a LaTeX paper, not an end-user product. These references are unavoidable for unambiguous acceptance criteria. The spec nevertheless avoids over-specifying algorithms (e.g., does not prescribe the internal structure of new modules) — those choices land in `/speckit.plan`.

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

**Clarifications resolved (2026-04-23)**: Q1 = destructive delete (FR-013); Q2 = two-layer policy — qualitative shape + ≤ 20 % per-bin relative error against FRFR-category empirical curves, MS-TCM no-worse-than `--standard-tcm` (FR-020, FR-021, SC-001); Q3 = JAX supports both dtypes with float64 default at 1e-10 and float32 opt-in at 1e-8 (FR-023, FR-032, User Story 2 Scenario 4, SC-006).

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [ ] No implementation details leak into specification

**Notes on "implementation details leak"**: see Content Quality note above. The `/speckit.clarify` step will not close this item — it is inherent to this feature's rewrite-of-the-codebase nature. Recording as an accepted trade-off.

## Notes

- Items marked incomplete require spec updates before `/speckit.clarify` or `/speckit.plan`
- The three open clarifications are load-bearing: Q1 affects whether the 001 feature's code survives, Q2 defines what "the model works" means at test time, Q3 affects cross-platform reproducibility math — all three need explicit user answers before planning
- The "implementation detail" flag in Content Quality and Feature Readiness is accepted as a structural feature of this rewrite (not a bug) and does not block the `/speckit.clarify` handoff
