"""C&Z 2025 hierarchical CMR for free recall (single-storyline reduction).

This is the authoritative implementation of Cornell & Zhang 2025's
hierarchical model for the free-recall case. Both Tier-1 (``likelihood.py``,
numpy) and Tier-2 (``jax_backend/hcmr_jax.py``, jax.numpy) delegate here
via the ``xp`` namespace. The math follows ``notes/CornZhan25.pdf`` Eqs
1-15 literally.

Canonical equations implemented:

    Eq 1  : c^item_i = ρ c^item_{i-1} + β_enc c^IN_enc
    Eq 2a : ΔM^FC_exp = c^item_{i-1} f_i^T    (context→item associations)
    Eq 2b : ΔM^CF_exp = f_i c^item_{i-1}^T    (item→context, scored at retrieval)
    Eq 3  : c^ret_j   = ρ c^ret_{j-1} + β_rec c^IN_rec
    Eq 4  : c^IN_rec  = (1-γ_fc) M^FC_pre · f_j + γ_fc M^FC_exp · f_j
    Eq 5  : a_j = M^CF_exp · c^ret_j          (NO primacy gradient φ_l)
    Eq 6  : p(j)      = softmax(k · a_j)
    Eq 7  : p_stop    = exp(-ε_d · a^nr / a^r)
    Eq 8  : c^list_i  = ρ c^list_{i-1} + β_list c^IN_enc
    Eq 9  : c^item_i  = c^list_i  at list boundaries
    Eq 15 : reinstate c^item ← c^list_beginning_of_list on item-level failure

For free recall (single list), "beginning-of-list context" is c^list_0 =
e_start (the orthogonal marker vector, unit-norm, orthogonal to every
item). C&Z §"the retrieval of the list-level context, once the item-level
context fails, is simply set to be the beginning-of-list context instead
of through a list-level retrieval process (i.e., removing Equations 12–13
and the parameter βpost)".

Parameter inventory (7 free for free-recall, matching C&Z 2025 Table 1):
    β_enc   (Eq 1)    = 0.679
    β_list  (Eq 8)    = 0.400
    β_rec   (Eq 3)    = 0.326
    β_rein  (Eq 14)   = 0.300  (unused in free-recall; reserved for FFR)
    γ_fc    (Eq 4)    = 0.315
    k       (Eq 6)    = 6.50
    ε_d     (Eq 7)    = 1.04

Primacy parameters φ_s, φ_d from base CMR are DROPPED — C&Z's hierarchical
extension replaces them with the reinstatement fallback (Eq 15), which
produces primacy in SPC without introducing a primacy gradient at encoding.

Marginalized likelihood: the phase-transition index T ∈ {0, ..., R} is
treated as a latent variable. The per-list log-likelihood marginalizes
over T via logsumexp:

    LL_list = logsumexp_{T=0..R}[
        Σ_{i=1..T} log(1 - p_stop_i^(1)) + log p(s_i | c_ret^(1)_i)
        + log p_stop_transition(c_ret^(1) after T drifts)
        + Σ_{i=T+1..R} log(1 - p_stop_i^(2)) + log p(s_i | c_ret^(2)_i)
        + log p_stop_final(c_ret^(2) after R-T drifts)
    ]
"""

from __future__ import annotations

from dataclasses import dataclass

from ms_tcm._likelihood_shared import (
    _is_jax,
    activation,
    at_add,
    at_set,
    c_IN_rec_of,
    compute_p_stop,
    drift_c_ret,
    log_softmax_masked,
    norm_preserving_rho,
)


# --- hyperparameter bundle -----------------------------------------------


@dataclass(frozen=True)
class CoreHyperparams:
    """Scalar C&Z 2025 free-recall parameters.

    Stored as Python floats so both backends cast into their preferred
    precision at the boundary (numpy float64 / configurable jax dtype).
    """

    beta_enc: float
    beta_list: float
    beta_rec: float
    beta_rein: float  # reserved for final-free-recall; unused in FR
    gamma_fc: float
    k: float
    epsilon_d: float

    @classmethod
    def from_model_parameters(cls, p) -> "CoreHyperparams":
        return cls(
            beta_enc=float(p.beta_enc),
            beta_list=float(p.beta_list),
            beta_rec=float(p.beta_rec),
            beta_rein=float(p.beta_rein),
            gamma_fc=float(p.gamma_fc),
            k=float(p.k),
            epsilon_d=float(p.epsilon_d),
        )


# --- encoding -------------------------------------------------------------


def run_encoding(
    hp: CoreHyperparams,
    W: int,
    xp,
    dtype,
):
    """Run the W-step C&Z encoding and return (c_item_traj, M_fc_exp, M_cf_exp).

    Inputs (basis-vector c^IN_enc = e_{t+1} — identity M^FC_pre). Returns:

    - ``c_item_traj`` : (W+1, d) item-level context after each encoding step.
      Row 0 = e_start; rows 1..W = c_item after each step.
    - ``M_fc_exp``    : (d, W) experimental context→item matrix (Eq 2a).
    - ``M_cf_exp``    : (W, d) experimental item→context matrix (Eq 2b).

    Note that free recall uses a single "storyline", so no storyline-level
    bookkeeping (M^SC, storyline returns, etc.) is needed — those are
    final-free-recall machinery we will add back when extending the model.
    """
    d = W + 1

    # Initial state: c_item = c_list = e_start.
    e_start = xp.zeros(d, dtype=dtype)
    e_start = at_set(e_start, 0, 1.0, xp)
    c_item = e_start
    c_list = e_start

    m_fc_exp = xp.zeros((d, W), dtype=dtype)
    m_cf_exp = xp.zeros((W, d), dtype=dtype)

    c_item_traj = xp.zeros((W + 1, d), dtype=dtype)
    c_item_traj = at_set(c_item_traj, 0, e_start, xp)

    if _is_jax(xp):
        import jax

        def step(carry, t):
            c_item, c_list, m_fc_exp, m_cf_exp = carry

            # Eq 2a,b: column/row updates using PRE-drift c_item.
            m_fc_exp = at_set(m_fc_exp, (slice(None), t), m_fc_exp[:, t] + c_item, xp)
            m_cf_exp = at_set(m_cf_exp, (t, slice(None)), m_cf_exp[t, :] + c_item, xp)

            # Basis-vector c^IN_enc = e_{t+1}: dot(c, e_{t+1}) = c[t+1].
            t_plus_1 = t + 1

            # Eq 1: item-level drift.
            dot_i = c_item[t_plus_1]
            rho_i = norm_preserving_rho(hp.beta_enc, dot_i, xp)
            c_item = rho_i * c_item
            c_item = at_add(c_item, t_plus_1, hp.beta_enc, xp)

            # Eq 8: list-level drift (slower rate).
            dot_l = c_list[t_plus_1]
            rho_l = norm_preserving_rho(hp.beta_list, dot_l, xp)
            c_list = rho_l * c_list
            c_list = at_add(c_list, t_plus_1, hp.beta_list, xp)

            return (c_item, c_list, m_fc_exp, m_cf_exp), c_item

        t_arr = xp.arange(W, dtype=xp.int32)
        init = (c_item, c_list, m_fc_exp, m_cf_exp)
        (_, _, m_fc_final, m_cf_final), c_item_steps = jax.lax.scan(step, init, t_arr)
        c_item_traj = xp.concatenate([e_start[None, :], c_item_steps], axis=0)
        return c_item_traj, m_fc_final, m_cf_final

    # numpy path
    for t in range(W):
        m_fc_exp = at_set(m_fc_exp, (slice(None), t), m_fc_exp[:, t] + c_item, xp)
        m_cf_exp = at_set(m_cf_exp, (t, slice(None)), m_cf_exp[t, :] + c_item, xp)
        t_plus_1 = t + 1
        dot_i = c_item[t_plus_1]
        rho_i = norm_preserving_rho(hp.beta_enc, dot_i, xp)
        c_item = rho_i * c_item
        c_item = at_add(c_item, t_plus_1, hp.beta_enc, xp)
        dot_l = c_list[t_plus_1]
        rho_l = norm_preserving_rho(hp.beta_list, dot_l, xp)
        c_list = rho_l * c_list
        c_list = at_add(c_list, t_plus_1, hp.beta_list, xp)
        c_item_traj = at_set(c_item_traj, t + 1, c_item, xp)

    return c_item_traj, m_fc_exp, m_cf_exp


# --- retrieval helpers (imported from _likelihood_shared) ----------------
# ``activation``, ``compute_p_stop``, ``drift_c_ret``, ``c_IN_rec_of`` are
# defined once in ``_likelihood_shared.py`` and shared with the MS-TCM
# extension (``_likelihood_core_mstcm.py``). See that module for the
# canonical definitions; this file just imports them at the top.


# --- per-list log-likelihood (marginalized over phase-transition) --------


def compute_list_log_likelihood(
    hp: CoreHyperparams,
    W: int,
    recall_sps,    # (R,) int: 1-based serial positions; 0 = padding/intrusion
    recall_mask,   # (R,) bool: True for valid recall slots
    xp,
    dtype,
):
    """End-to-end per-list log-likelihood under C&Z 2025 free-recall.

    The observed recall sequence s_1, ..., s_R (from ``recall_sps`` at
    positions where ``recall_mask`` is True) is marginalized over the
    latent phase-transition index T ∈ {0, ..., R}:

        LL_list = logsumexp_{T=0..R} LL_T

    where LL_T is the log-likelihood conditional on the first T recalls
    being in phase 1 (cued by c_item_end, drifted via β_rec) and the
    remaining R-T recalls being in phase 2 (cued by e_start, drifted
    via β_rec). The transition itself contributes log(p_stop) at the
    c_ret value after T phase-1 drifts; the final stop (after the last
    phase-2 recall) contributes log(p_stop) again.

    This function uses a Python/numpy-style loop when ``xp`` is numpy;
    under jax.numpy the loop is driven by ``jax.lax.scan`` with vmap
    over T for parallel evaluation. Both produce bit-identical results
    up to float rounding.
    """
    d = W + 1
    c_item_traj, m_fc_exp, m_cf_exp = run_encoding(hp, W, xp, dtype)

    # Phase 1 starts with c_ret = c_item_end (end-of-list cue).
    c_ret_init_phase1 = c_item_traj[W]
    # Phase 2 starts with c_ret = e_start (beginning-of-list context).
    c_ret_init_phase2 = xp.zeros(d, dtype=dtype)
    c_ret_init_phase2 = at_set(c_ret_init_phase2, 0, 1.0, xp)

    # Determine R = number of valid recalls.
    # recall_mask[i] = True iff recall i is valid (not padding).
    R_total = recall_sps.shape[0]

    # For each phase-1 length T in {0, 1, ..., R_total}, compute LL_T.
    # We do this by simulating both phase 1 and phase 2 forward, recording
    # per-step "running log-prob given we haven't transitioned yet" and
    # "running log-prob given we have already transitioned at step t=K".
    # Then LL_T = phase1_ll(T) + log(p_stop at c_ret_phase1 after T drifts)
    #           + phase2_ll(R-T, with R_valid = count of valid entries)
    #           + log(p_stop_final at c_ret_phase2 after R-T drifts)
    #
    # Because T ranges over all recall counts, padding rows (valid=False)
    # must be ignored. We precompute forward passes of phase 1 and phase 2
    # producing length-(R_total+1) arrays of c_ret states and cumulative
    # LL up to each step; the "after T drifts" state is c_ret_trajectory[T].

    # --- Phase-1 forward pass ---
    # Tracks state: c_ret. At each step, if recall_mask[i] is True and
    # sp[i] is a valid 1-based index:
    #   contrib_i = log(1 - p_stop(c_ret)) + log_softmax[sp-1]
    #   c_ret ← drift(c_ret, c_IN_rec(sp-1))
    # Padded (valid=False) steps contribute 0 and leave c_ret untouched.

    def _forward_phase(c_ret_init):
        """Roll c_ret and already_mask forward through all R_total slots.

        Returns:
        - cum_ll     : (R_total+1,) cumulative LL through step-i-inclusive
                       (counting only valid non-padding slots).
        - c_ret_traj : (R_total+1, d) c_ret at the START of each slot.
        - mask_traj  : (R_total+1, W) already-recalled mask at START of each slot.

        Contributions include ``log(1 - p_stop)`` at the pre-drift c_ret AND
        ``log softmax_masked(k·a)`` at the observed sp. The transition
        term ``log p_stop`` is NOT included here — the caller adds it for
        each T hypothesis using ``c_ret_traj[T]`` and ``mask_traj[T]``.

        Masking: at slot i, the already_mask reflects valid recalls
        observed in slots 0..i-1; padding slots leave it unchanged.
        """
        if _is_jax(xp):
            import jax

            def step(carry, slot):
                c_ret, cum, already_mask = carry
                sp, valid = slot
                idx = xp.clip(sp - 1, 0, W - 1)
                # Repeats are treated as noise (no score contribution, no
                # stopping contribution, no state update). This matches the
                # pre-refactor v6 likelihood's repeat-tolerance behavior.
                is_repeat = already_mask[idx]
                countable = valid & ~is_repeat
                p_stop = compute_p_stop(
                    m_cf_exp, c_ret, already_mask, hp.epsilon_d, xp,
                )
                log_1m = xp.log(xp.maximum(1.0 - p_stop, 1e-300))
                a = activation(m_cf_exp, c_ret, xp)
                log_p = log_softmax_masked(hp.k * a, ~already_mask, xp)
                contrib = xp.where(countable, log_1m + log_p[idx], 0.0)
                new_cum = cum + contrib
                # drift
                c_IN = c_IN_rec_of(idx, m_fc_exp, hp.gamma_fc, d, xp, dtype)
                c_ret_drifted = drift_c_ret(c_ret, c_IN, hp.beta_rec, xp)
                new_c_ret = xp.where(countable, c_ret_drifted, c_ret)
                new_mask = xp.where(
                    countable,
                    at_set(already_mask, idx, True, xp),
                    already_mask,
                )
                return (new_c_ret, new_cum, new_mask), (new_c_ret, new_cum, new_mask)

            init = (c_ret_init, xp.asarray(0.0, dtype=dtype),
                    xp.zeros(W, dtype=bool))
            _, (c_ret_1toR, cum_1toR, mask_1toR) = jax.lax.scan(
                step, init, (recall_sps, recall_mask),
            )
            c_ret_traj = xp.concatenate(
                [c_ret_init[None, :], c_ret_1toR], axis=0,
            )
            cum_ll = xp.concatenate(
                [xp.zeros((1,), dtype=dtype), cum_1toR], axis=0,
            )
            mask_traj = xp.concatenate(
                [xp.zeros((1, W), dtype=bool), mask_1toR], axis=0,
            )
            return cum_ll, c_ret_traj, mask_traj

        # numpy path
        c_ret = (c_ret_init.copy() if hasattr(c_ret_init, 'copy')
                 else xp.array(c_ret_init))
        cum = 0.0
        cum_ll = xp.zeros((R_total + 1,), dtype=dtype)
        c_ret_traj = xp.zeros((R_total + 1, d), dtype=dtype)
        mask_traj = xp.zeros((R_total + 1, W), dtype=bool)
        c_ret_traj = at_set(c_ret_traj, 0, c_ret, xp)
        already_mask = xp.zeros(W, dtype=bool)
        for i in range(R_total):
            sp = int(recall_sps[i])
            valid = bool(recall_mask[i])
            idx = max(0, min(W - 1, sp - 1))
            is_repeat = bool(already_mask[idx])
            countable = valid and not is_repeat
            p_stop = compute_p_stop(
                m_cf_exp, c_ret, already_mask, hp.epsilon_d, xp,
            )
            log_1m = float(xp.log(max(1.0 - float(p_stop), 1e-300)))
            a = activation(m_cf_exp, c_ret, xp)
            log_p = log_softmax_masked(hp.k * a, ~already_mask, xp)
            contrib = (log_1m + float(log_p[idx])) if countable else 0.0
            cum = cum + contrib
            if countable:
                c_IN = c_IN_rec_of(idx, m_fc_exp, hp.gamma_fc, d, xp, dtype)
                c_ret = drift_c_ret(c_ret, c_IN, hp.beta_rec, xp)
                already_mask = at_set(already_mask, idx, True, xp)
            cum_ll = at_set(cum_ll, i + 1, cum, xp)
            c_ret_traj = at_set(c_ret_traj, i + 1, c_ret, xp)
            mask_traj = at_set(mask_traj, i + 1, already_mask, xp)
        return cum_ll, c_ret_traj, mask_traj

    cum_ll_phase1, c_ret_traj_phase1, mask_traj_phase1 = _forward_phase(
        c_ret_init_phase1,
    )
    cum_ll_phase2, c_ret_traj_phase2, mask_traj_phase2 = _forward_phase(
        c_ret_init_phase2,
    )

    # Wait — phase 2 starts with c_ret = e_start AND already_mask reflecting
    # the recalls that happened in phase 1. So a separate phase-2 pass for
    # each T would differ by the initial already_mask. But the already_mask
    # evolution across slots is INVARIANT to T (it's the observed sequence's
    # cumulative recalled set). So mask_traj_phase1 == mask_traj_phase2 —
    # both reflect the same observed cumulative recalls. Good: we can use
    # mask_traj_phase1 as THE mask evolution for both phases.
    #
    # What DOES differ across T is c_ret: under T=t, the first t slots use
    # phase-1 c_ret (drifts from c_item_end), and slots t..R use phase-2
    # c_ret (drifts from e_start). Because our two forward passes compute
    # c_ret assuming each phase is active for ALL slots, we have:
    #   For T=t:
    #     slots 0..t-1 use c_ret_traj_phase1[0..t-1]   (correct)
    #     slots t..R-1 use c_ret_traj_phase2[0..R-1-t] (but our phase-2
    #       forward started from slot 0 with e_start and drifts from
    #       there, which DOESN'T match — the correct phase-2 c_ret for
    #       slot t (the first phase-2 slot) is e_start, and for slot t+k
    #       is e_start drifted by the RECALLS observed at slots t, t+1,
    #       ..., t+k-1. That matches c_ret_traj_phase2[k].)
    #
    # So the phase-2 contribution for slots t..R-1 uses c_ret_traj_phase2
    # SHIFTED — specifically slot (t+k) in the observed sequence uses
    # c_ret_traj_phase2[k]. This is delicate.
    #
    # But cum_ll_phase2[i] assumes slots 0..i-1 were ALL phase-2. If
    # under T=t the first t slots are phase-1 and slots t..R-1 are
    # phase-2, we want:
    #   phase1_contrib = cum_ll_phase1[t]
    #   phase2_contrib = cum_ll_phase2[R - t]   (the first R-t phase-2
    #                      contributions, computed from c_ret=e_start
    #                      drifting through the observed recalls at
    #                      slots t, t+1, ..., R-1)
    # But THAT requires the phase-2 forward to process the SAME observed
    # recall sequence shifted — i.e., recall_sps[t], recall_sps[t+1], ...
    # Running a separate phase-2 forward for each t is O(R^2 W). That's
    # tractable for Kahana scale (R ≤ 10, so 55 passes per list).
    #
    # Actually, the observation above is key: once we're in phase 2, the
    # c_ret evolution depends only on the OBSERVED RECALL SEQUENCE FROM
    # THAT POINT ON. Because phase-2 starts fresh at e_start, then drifts
    # toward c_IN_rec of whichever item was observed, the cumulative
    # contribution depends on recall_sps[t:] in order. So we CANNOT reuse
    # cum_ll_phase2 unless we re-run phase-2 with each different starting
    # slot. That's R+1 re-runs. Let's do it.

    # Re-run phase-2 for each starting slot t in {0, ..., R_total}.
    # For t=R_total, phase-2 is empty (0 recalls), final c_ret = e_start.
    # For t=0, phase-2 is full (all R_total slots).

    # Prepare arrays to hold phase-2 results per starting-slot t.
    ll_per_T = _compute_ll_per_T(
        R_total=R_total, W=W, d=d,
        c_ret_traj_phase1=c_ret_traj_phase1,
        cum_ll_phase1=cum_ll_phase1,
        mask_traj=mask_traj_phase1,
        recall_sps=recall_sps, recall_mask=recall_mask,
        m_fc_exp=m_fc_exp, m_cf_exp=m_cf_exp,
        hp=hp, xp=xp, dtype=dtype,
        c_ret_init_phase2=c_ret_init_phase2,
    )

    # logsumexp over T.
    m = xp.max(ll_per_T)
    return m + xp.log(xp.sum(xp.exp(ll_per_T - m)))


def _compute_ll_per_T(
    *, R_total, W, d,
    c_ret_traj_phase1, cum_ll_phase1, mask_traj,
    recall_sps, recall_mask,
    m_fc_exp, m_cf_exp, hp, xp, dtype,
    c_ret_init_phase2,
):
    """Compute LL for each phase-transition index T ∈ {0, ..., R_total}.

    Phase-2 starts fresh at e_start for each T and processes the observed
    recalls at slots T, T+1, ..., R_total-1 using the observed-sequence
    already_mask trajectory.

    Implementation:
    - Under JAX: vmaps a single-T phase-2 forward across all T values; each
      single-T forward is a ``lax.scan`` over R_total slots gated by a
      ``slot_idx >= T`` mask. JIT-compile cost is O(W) regardless of R.
    - Under numpy: explicit double loop over (T, slot). Slow but correct.
    """
    if _is_jax(xp):
        return _compute_ll_per_T_jax(
            R_total=R_total, W=W, d=d,
            c_ret_traj_phase1=c_ret_traj_phase1,
            cum_ll_phase1=cum_ll_phase1, mask_traj=mask_traj,
            recall_sps=recall_sps, recall_mask=recall_mask,
            m_fc_exp=m_fc_exp, m_cf_exp=m_cf_exp,
            hp=hp, xp=xp, dtype=dtype,
            c_ret_init_phase2=c_ret_init_phase2,
        )

    # numpy: straightforward double loop.
    ll_per_T = xp.zeros((R_total + 1,), dtype=dtype)
    for T in range(R_total + 1):
        phase1_ll = cum_ll_phase1[T]
        c_ret_at_trans = c_ret_traj_phase1[T]
        mask_at_trans = mask_traj[T]
        p_stop_trans = compute_p_stop(
            m_cf_exp, c_ret_at_trans, mask_at_trans, hp.epsilon_d, xp,
        )
        log_trans = xp.log(xp.maximum(p_stop_trans, 1e-300))

        c_ret = c_ret_init_phase2
        mask = mask_at_trans
        phase2_ll = xp.asarray(0.0, dtype=dtype)
        for i in range(T, R_total):
            sp = recall_sps[i]
            valid = recall_mask[i]
            idx = xp.clip(sp - 1, 0, W - 1)
            is_repeat = mask[idx]
            count_this = valid & ~is_repeat
            p_stop_i = compute_p_stop(
                m_cf_exp, c_ret, mask, hp.epsilon_d, xp,
            )
            log_1m_i = xp.log(xp.maximum(1.0 - p_stop_i, 1e-300))
            a = activation(m_cf_exp, c_ret, xp)
            log_p_i = log_softmax_masked(hp.k * a, ~mask, xp)
            contrib_i = xp.where(count_this, log_1m_i + log_p_i[idx], 0.0)
            phase2_ll = phase2_ll + contrib_i
            c_IN = c_IN_rec_of(idx, m_fc_exp, hp.gamma_fc, d, xp, dtype)
            c_ret_drifted = drift_c_ret(c_ret, c_IN, hp.beta_rec, xp)
            c_ret = xp.where(count_this, c_ret_drifted, c_ret)
            mask = xp.where(
                count_this, at_set(mask, idx, True, xp), mask,
            )

        p_stop_final = compute_p_stop(m_cf_exp, c_ret, mask, hp.epsilon_d, xp)
        log_final = xp.log(xp.maximum(p_stop_final, 1e-300))
        ll_T = phase1_ll + log_trans + phase2_ll + log_final
        ll_per_T = at_set(ll_per_T, T, ll_T, xp)

    return ll_per_T


def _compute_ll_per_T_jax(
    *, R_total, W, d,
    c_ret_traj_phase1, cum_ll_phase1, mask_traj,
    recall_sps, recall_mask,
    m_fc_exp, m_cf_exp, hp, xp, dtype,
    c_ret_init_phase2,
):
    """JAX path for ``_compute_ll_per_T``: O(W) compile cost via scan+vmap.

    For each candidate T, runs a single ``lax.scan`` over R_total slots.
    The scan body checks ``slot_idx >= T`` to decide whether the slot
    contributes (phase 2 active) or is skipped (still in phase 1).
    """
    import jax

    # Helper: phase-2 forward for a SINGLE T value.
    # State: (c_ret, mask, cum_phase2_ll).
    def phase2_for_T(T):
        # Initial mask = phase-1's accumulated mask AT slot T.
        init_mask = mask_traj[T]
        init = (
            c_ret_init_phase2,
            init_mask,
            xp.asarray(0.0, dtype=dtype),
        )

        # Scan over all R_total slots; gate by `slot_idx >= T`. Repeats
        # are treated as noise (skip score+state-update like v6 core).
        def step(carry, slot_data):
            c_ret, mask, cum = carry
            slot_idx, sp, valid = slot_data
            in_phase2 = slot_idx >= T
            idx = xp.clip(sp - 1, 0, W - 1)
            is_repeat = mask[idx]
            count_this = valid & in_phase2 & ~is_repeat

            p_stop = compute_p_stop(
                m_cf_exp, c_ret, mask, hp.epsilon_d, xp,
            )
            log_1m = xp.log(xp.maximum(1.0 - p_stop, 1e-300))
            a = activation(m_cf_exp, c_ret, xp)
            log_p = log_softmax_masked(hp.k * a, ~mask, xp)
            contrib = xp.where(count_this, log_1m + log_p[idx], 0.0)
            new_cum = cum + contrib

            # Drift & mask update only on counted slots.
            c_IN = c_IN_rec_of(idx, m_fc_exp, hp.gamma_fc, d, xp, dtype)
            c_ret_drifted = drift_c_ret(c_ret, c_IN, hp.beta_rec, xp)
            new_c_ret = xp.where(count_this, c_ret_drifted, c_ret)
            new_mask = xp.where(
                count_this, at_set(mask, idx, True, xp), mask,
            )
            return (new_c_ret, new_mask, new_cum), None

        slot_idxs = xp.arange(R_total, dtype=xp.int32)
        (c_ret_final, mask_final, cum_final), _ = jax.lax.scan(
            step, init, (slot_idxs, recall_sps, recall_mask),
        )

        # Phase-1 contribution + transition log_p_stop.
        phase1_ll = cum_ll_phase1[T]
        c_ret_at_trans = c_ret_traj_phase1[T]
        mask_at_trans = mask_traj[T]
        p_stop_trans = compute_p_stop(
            m_cf_exp, c_ret_at_trans, mask_at_trans, hp.epsilon_d, xp,
        )
        log_trans = xp.log(xp.maximum(p_stop_trans, 1e-300))

        # Final stop on phase-2's terminal c_ret + mask.
        p_stop_final = compute_p_stop(
            m_cf_exp, c_ret_final, mask_final, hp.epsilon_d, xp,
        )
        log_final = xp.log(xp.maximum(p_stop_final, 1e-300))

        return phase1_ll + log_trans + cum_final + log_final

    Ts = xp.arange(R_total + 1, dtype=xp.int32)
    return jax.vmap(phase2_for_T)(Ts)


# --- numpy entry point ---------------------------------------------------


def compute_list_log_likelihood_numpy(
    params,
    recall_sps,
    recall_mask,
    W: int,
) -> float:
    """Numpy-dtype entry point (Tier-1 and oracle use this)."""
    import numpy as np

    hp = CoreHyperparams.from_model_parameters(params)
    ll = compute_list_log_likelihood(
        hp,
        W=W,
        recall_sps=np.asarray(recall_sps, dtype=np.int64),
        recall_mask=np.asarray(recall_mask, dtype=bool),
        xp=np, dtype=np.float64,
    )
    return float(ll)


# --- simulator (sample recalls from the generative model) ----------------


def simulate_recalls(
    params,
    W: int,
    rng,  # numpy random.Generator
    *,
    max_recalls: int | None = None,
) -> list[int]:
    """Sample a recall sequence from the C&Z hierarchical model (FR mode).

    Implements the forward generative process literally:

        phase = 1
        c_ret = c_item_traj[W]   (end-of-list cue)
        while not terminated:
            compute a = M^CF_exp · c_ret; mask already_recalled
            compute p_stop via Eq 7
            if rng < p_stop:
                if phase == 1:
                    c_ret = e_start   (reinstate to beginning-of-list)
                    phase = 2
                    continue
                else:
                    break   (terminate)
            # else draw a recall
            i ~ Categorical(softmax(k · a_masked))
            record i; already_recalled.add(i)
            c_ret <- drift(c_ret, β_rec, c^IN_rec(i))

    Returns the list of 1-based serial positions produced.
    """
    import numpy as np

    hp = CoreHyperparams.from_model_parameters(params)
    d = W + 1

    c_item_traj, m_fc_exp, m_cf_exp = run_encoding(hp, W, np, np.float64)

    c_ret = c_item_traj[W].copy()
    phase = 1
    already_mask = np.zeros(W, dtype=bool)
    recalls: list[int] = []
    cap = max_recalls if max_recalls is not None else 3 * W

    while len(recalls) < cap:
        p_stop = float(compute_p_stop(m_cf_exp, c_ret, already_mask, hp.epsilon_d, np))
        if rng.random() < p_stop:
            if phase == 1:
                c_ret = np.zeros(d, dtype=np.float64)
                c_ret[0] = 1.0  # e_start
                phase = 2
                continue
            else:
                break
        a = activation(m_cf_exp, c_ret, np)
        log_p = log_softmax_masked(hp.k * a, ~already_mask, np)
        p = np.exp(log_p - log_p.max())
        p = p / p.sum()
        # Mask: anything already recalled should have been driven to ~0 by
        # the softmax mask; re-zero defensively.
        p = np.where(already_mask, 0.0, p)
        p_sum = p.sum()
        if p_sum <= 0:
            break
        p = p / p_sum
        idx = int(rng.choice(W, p=p))
        recalls.append(idx + 1)  # 1-based SP
        already_mask[idx] = True
        c_IN = c_IN_rec_of(idx, m_fc_exp, hp.gamma_fc, d, np, np.float64)
        c_ret = drift_c_ret(c_ret, c_IN, hp.beta_rec, np)

    return recalls
