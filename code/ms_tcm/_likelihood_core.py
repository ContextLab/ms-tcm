"""Single-source-of-truth likelihood core for MS-TCM v6 (Constitution II).

Both Tier-1 (``likelihood.py``) and Tier-2 (``jax_backend/hcmr_jax.py``)
delegate to this module. The math is parameterized by an ``xp`` namespace
(``numpy`` or ``jax.numpy``) and a ``scan_fn`` for the recurrence; the
two backends differ ONLY in which array library they pass in, not in
which formulas they apply.

v6 canonical source: ``notes/two_level_cmr_v6.pdf``. Equations cited:

- Eq 1: c^item_i = ρ_enc · c^item_{i-1} + β_enc · c^IN_i
- Eq 2: c^story = ρ_story · c^story_{j-1} + β_story · c^IN_i (storyline)
- Eq 5: ΔM^SC = g_s · (c^story_out)^T (cache on switch)
- Eq 6: c^story_new = λ · c~^story_{s(i)} + (1 - λ) · c^story_prev
- Eq 7: c^item ← c^story after switch/return
- Eq 3 (C&Z 2025): c^ret drifts toward c^item_recalled at rate β_rec
- Eq 7 (C&Z 2025): p_stop = exp(-ε_d · a^nr / a^r), sums of |a|
- Eq 8-9: a = (M^IC)^T · c_cue; P(j|cue) = softmax(k · a)

Design: every array op goes through ``xp.*``; in-place updates go through
the ``at_set`` / ``at_add`` helpers which dispatch on the namespace.
Under numpy, the driver function uses a Python loop; under jax.numpy, it
uses ``jax.lax.scan`` — but the per-step function body is identical.

Parity property (enforced by ``test_backend_parity.py``):
    compute_list_log_likelihood(hp, ..., xp=numpy)
        == compute_list_log_likelihood(hp, ..., xp=jnp (float64))
within 1e-10 absolute tolerance on the full FRFR-category dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


# --- xp dispatch helpers -------------------------------------------------
#
# These tiny helpers paper over the one place numpy and jax.numpy don't
# share an API: in-place updates. numpy uses ``x[i] = v``; jax.numpy uses
# ``x = x.at[i].set(v)`` because JAX arrays are immutable.


def _is_jax(xp: Any) -> bool:
    """Return True if ``xp`` is jax.numpy (detected via ``__name__``)."""
    name = getattr(xp, "__name__", "")
    return name == "jax.numpy"


def at_set(x, idx, value, xp):
    """Functional scatter: return a new array with ``x[idx] = value``.

    Works for both 1-D scalar indices (``x[idx] = value``) and tuple
    indices (``x[idx0, idx1] = value``).
    """
    if _is_jax(xp):
        return x.at[idx].set(value)
    x = x.copy()
    x[idx] = value
    return x


def at_add(x, idx, delta, xp):
    """Functional scatter-add: return a new array with ``x[idx] += delta``."""
    if _is_jax(xp):
        return x.at[idx].add(delta)
    x = x.copy()
    x[idx] = x[idx] + delta
    return x


def log_softmax_masked(scaled, mask, xp):
    """Numerically stable log-softmax with a boolean keep-mask.

    ``mask[i] = True`` keeps item i in the competition; ``mask[i] = False``
    drives its log-probability to a large negative sentinel (-1e30) so the
    caller can detect that reading it was an error (we never do — callers
    always index valid items only).

    This is the single softmax implementation — no scipy, no jax.nn
    variant. Bit-identical under numpy and jax.numpy for the same dtype.
    """
    # Replace masked positions with a very negative sentinel BEFORE
    # computing the max. This keeps the max/shift math well-defined when
    # the only unmasked activations are themselves strongly negative.
    NEG = -1.0e30
    scaled = xp.where(mask, scaled, NEG)
    m = xp.max(scaled)
    shifted = scaled - m
    exp_s = xp.exp(shifted)
    # Zero out masked positions in the exponentiated sum (where exp(NEG-m)
    # is already tiny, but we want it to be exactly 0 so log is well-defined).
    exp_s = xp.where(mask, exp_s, 0.0)
    denom = xp.sum(exp_s)
    # Guard against all-masked cases (denom=0): when the entire list is
    # already recalled, the caller's ``countable`` flag is False so the
    # value we return here is never read, but log(0) still raises a
    # numpy RuntimeWarning. Floor denom at 1e-300 to keep the op clean.
    return scaled - m - xp.log(xp.maximum(denom, 1e-300))


def norm_preserving_rho(beta, dot, xp):
    """ρ so ||ρ · c_prev + β · c_in|| = 1 for unit-norm c_prev, c_in.

    Matches ``ms_tcm.drift._norm_preserving_rho`` exactly.
    """
    inner = 1.0 + beta * beta * (dot * dot - 1.0)
    # xp.maximum handles the numpy/jax convention divergence (np.maximum is
    # elementwise, jnp.maximum is elementwise — same semantics here).
    safe = xp.maximum(inner, 0.0)
    return xp.sqrt(safe) - beta * dot


# --- hyperparameter bundle -----------------------------------------------


@dataclass(frozen=True)
class CoreHyperparams:
    """Scalar model parameters passed to the core.

    Stored as Python floats so numpy and jax drivers can each cast into
    their preferred precision at the boundary (float64 numpy / configurable
    jax dtype).
    """

    beta_enc: float
    beta_story: float
    gamma_fc: float  # currently unused (identity M^FC_pre); preserved for future
    k: float
    lambda_reinstate: float
    beta_rec: float
    epsilon_d: float
    standard_tcm: bool
    # Primacy scaling parameters (v6 §3 / Polyn et al. 2009 Eq 7).
    # φ and ψ differ based on standard_tcm (see ms_tcm.hcmr.encode).
    phi: float
    psi: float

    @classmethod
    def from_model_parameters(cls, p) -> "CoreHyperparams":
        """Build a CoreHyperparams from a ``ModelParameters`` instance."""
        if p.standard_tcm:
            phi, psi = 30.0, 0.8
        else:
            phi, psi = 1.5, 0.5
        return cls(
            beta_enc=float(p.beta_enc),
            beta_story=float(p.beta_story),
            gamma_fc=float(p.gamma_fc),
            k=float(p.k),
            lambda_reinstate=float(p.lambda_reinstate),
            beta_rec=float(p.beta_rec),
            epsilon_d=float(p.epsilon_d),
            standard_tcm=bool(p.standard_tcm),
            phi=phi,
            psi=psi,
        )


# --- encoding (per-step; scan-friendly) ----------------------------------


def make_e_start(d: int, xp, dtype):
    """Return e_start = (1, 0, ..., 0) as a shape-(d,) array of ``dtype``."""
    z = xp.zeros(d, dtype=dtype)
    return at_set(z, 0, 1.0, xp)


def make_basis(d: int, i, xp, dtype):
    """Return e_i as a shape-(d,) array (one-hot at position i)."""
    z = xp.zeros(d, dtype=dtype)
    return at_set(z, i, 1.0, xp)


def encode_step(
    state,
    step_inputs,
    hp: CoreHyperparams,
    W: int,
    K: int,
    xp,
    dtype,
):
    """One encoding step of the v6 hierarchical CMR.

    Inputs:
        state: (c_item, c_story_stack, m_ic, m_sc, seen, prev_cat)
            c_item         (d,) current item-level context
            c_story_stack  (K, d) current story context per storyline
            m_ic           (d, W) accumulating M^IC
            m_sc           (K, d) cached storyline contexts
            seen           (K,) bool; True iff storyline has been encoded before
            prev_cat       int scalar; previous step's storyline index
                           (-1 at step 0)
        step_inputs: (t, cat_idx)
            t       int scalar (0..W-1)
            cat_idx int scalar (0..K-1)

    Returns:
        new_state of the same shape.

    The c^IN is the orthogonal one-hot e_{t+1} (identity M^FC_pre, v6 §1.5
    Option 1), so all dot products with c_item / c_story collapse to a
    single element lookup at index t+1.
    """
    c_item, c_story_stack, m_ic, m_sc, seen, prev_cat = state
    t, cat_idx = step_inputs

    # Boundary detection.
    switch = (prev_cat >= 0) & (cat_idx != prev_cat)
    cat_return = switch & seen[cat_idx]

    if not hp.standard_tcm:
        # --- Storyline switch: cache outgoing storyline to M^SC (v6 Eq 5).
        # m_sc[prev_cat] += c_story_stack[prev_cat].
        # We compute the cached-update unconditionally then pick via where.
        cached_row_candidate = m_sc[prev_cat] + c_story_stack[prev_cat]
        m_sc = xp.where(
            switch,
            at_set(m_sc, prev_cat, cached_row_candidate, xp),
            m_sc,
        )

        # --- Storyline return (v6 Eq 6): blend cached context into active.
        blended = (hp.lambda_reinstate * m_sc[cat_idx]
                   + (1.0 - hp.lambda_reinstate) * c_story_stack[cat_idx])
        nn = xp.sqrt(xp.sum(blended * blended))
        blended_n = xp.where(nn > 1e-12, blended / xp.maximum(nn, 1e-30), blended)
        c_story_stack = xp.where(
            cat_return,
            at_set(c_story_stack, cat_idx, blended_n, xp),
            c_story_stack,
        )

        # --- Eq 7: on (switch OR return), sync c_item to active storyline.
        c_item = xp.where(switch, c_story_stack[cat_idx], c_item)

    # Basis-vector c^IN: e_{t+1}; dot(x, c^IN) = x[t+1].
    t_plus_1 = t + 1

    # --- Storyline drift (v6 Eq 2).
    if not hp.standard_tcm:
        c_story_active = c_story_stack[cat_idx]
        dot_s = c_story_active[t_plus_1]
        rho_s = norm_preserving_rho(hp.beta_story, dot_s, xp)
        c_story_drifted = rho_s * c_story_active
        c_story_drifted = at_add(c_story_drifted, t_plus_1, hp.beta_story, xp)
        c_story_stack = at_set(c_story_stack, cat_idx, c_story_drifted, xp)

    # --- Item-level drift (v6 Eq 1).
    dot_i = c_item[t_plus_1]
    rho_i = norm_preserving_rho(hp.beta_enc, dot_i, xp)
    c_item = rho_i * c_item
    c_item = at_add(c_item, t_plus_1, hp.beta_enc, xp)

    # --- M^IC update: M^IC[:, t] += primacy(t) · c_item.
    primacy = 1.0 + hp.phi * xp.exp(-hp.psi * xp.asarray(t, dtype=dtype))
    # at_add with tuple index (:, t) doesn't map cleanly across backends;
    # instead, add primacy * c_item to the t-th column directly.
    m_ic_col = m_ic[:, t] + primacy * c_item
    m_ic = at_set(m_ic, (slice(None), t), m_ic_col, xp)

    # --- Bookkeeping.
    seen = at_set(seen, cat_idx, True, xp)
    prev_cat = cat_idx

    return (c_item, c_story_stack, m_ic, m_sc, seen, prev_cat)


def run_encoding(
    hp: CoreHyperparams,
    cat_indices,
    W: int,
    K: int,
    xp,
    dtype,
):
    """Run the W-step encoding loop and return (c_item_traj, m_ic).

    Under numpy, uses a Python ``for`` loop. Under jax.numpy, uses
    ``jax.lax.scan`` (which compiles to a single XLA op). The per-step
    function (``encode_step``) is identical in both cases.
    """
    d = W + 1
    # Initial state.
    e_start = make_e_start(d, xp, dtype)
    c_item = e_start
    c_story_stack = xp.broadcast_to(e_start, (K, d))
    # ``broadcast_to`` in numpy returns a read-only view; make it writable.
    if not _is_jax(xp):
        c_story_stack = xp.array(c_story_stack)  # materialize a fresh array
    m_ic = xp.zeros((d, W), dtype=dtype)
    m_sc = xp.zeros((K, d), dtype=dtype)
    seen = xp.zeros(K, dtype=bool)
    prev_cat = xp.asarray(-1, dtype=xp.int32 if _is_jax(xp) else None)
    # For numpy, use a plain int; the xp.int32 requirement is a JAX-ism.
    if not _is_jax(xp):
        prev_cat = -1  # plain Python int works under numpy

    c_item_traj = xp.zeros((W + 1, d), dtype=dtype)
    c_item_traj = at_set(c_item_traj, 0, e_start, xp)

    if _is_jax(xp):
        import jax

        def scan_body(state, inp):
            new_state = encode_step(state, inp, hp, W, K, xp, dtype)
            return new_state, new_state[0]  # yield c_item

        t_arr = xp.arange(W, dtype=xp.int32)
        init_state = (c_item, c_story_stack, m_ic, m_sc, seen, prev_cat)
        (c_item_final, _, m_ic_final, _, _, _), c_item_steps = jax.lax.scan(
            scan_body, init_state, (t_arr, cat_indices),
        )
        c_item_traj = xp.concatenate([e_start[None, :], c_item_steps], axis=0)
        return c_item_traj, m_ic_final

    # numpy path: explicit Python loop.
    state = (c_item, c_story_stack, m_ic, m_sc, seen, prev_cat)
    for t in range(W):
        state = encode_step(
            state,
            (t, int(cat_indices[t])),
            hp, W, K, xp, dtype,
        )
        c_item_traj = at_set(c_item_traj, t + 1, state[0], xp)
    _, _, m_ic_final, _, _, _ = state
    return c_item_traj, m_ic_final


# --- retrieval + stopping ------------------------------------------------


def activation(m_ic, c_cue, xp):
    """a = (M^IC)^T @ c_cue. Shape: (W,)."""
    return m_ic.T @ c_cue


def stopping_log_terms(
    m_ic, c_ret, already_mask, hp: CoreHyperparams, xp,
):
    """Return (p_stop, log(1 - p_stop), log(p_stop)).

    Implements C&Z 2025 Eq 7 with |a| sums (matches
    ``HierarchicalCMRModel.stopping_prob_after_recalls``).

    Numerical guards:
    - ``a_r = 0`` (no recalls yet, or all-zero activations on recalled
      items): p_stop is set to 0 (the stopping rule is undefined without
      at least one recall's worth of activation). log(1-p_stop) = 0;
      log(p_stop) = log(1e-300) as a sentinel (caller should not use it
      in that case).
    """
    a = activation(m_ic, c_ret, xp)
    a_abs = xp.abs(a)
    a_r = xp.sum(xp.where(already_mask, a_abs, 0.0))
    a_nr = xp.sum(xp.where(already_mask, 0.0, a_abs))
    ratio = a_nr / xp.maximum(a_r, 1e-30)
    p_stop_raw = xp.exp(-hp.epsilon_d * ratio)
    # When a_r is effectively zero, force p_stop = 0.
    p_stop = xp.where(a_r > 0.0, p_stop_raw, 0.0)
    # Numerically guarded logs.
    log_1m = xp.log(xp.maximum(1.0 - p_stop, 1e-300))
    log_p = xp.log(xp.maximum(p_stop, 1e-300))
    return p_stop, log_1m, log_p


def drift_retrieval_context(c_ret, beta_rec: float, c_target, xp):
    """C&Z 2025 Eq 3: c_ret <- ρ_rec · c_ret + β_rec · c_target.

    Matches ``ms_tcm.retrieval.drift_retrieval_context`` exactly.
    """
    dot = xp.sum(c_ret * c_target)
    rho = norm_preserving_rho(beta_rec, dot, xp)
    return rho * c_ret + beta_rec * c_target


def recall_step(
    state,
    step_inputs,
    hp: CoreHyperparams,
    W: int,
    c_item_traj,
    m_ic,
    xp,
    dtype,
):
    """One recall step: contributes log(1 - p_stop) + log(p[sp]) to log_l.

    This is the single scan-friendly body that both backends use. The
    ordering of operations matches ``ms_tcm.likelihood.list_log_likelihood``:

    1. If this row is valid AND NOT a repeat AND there is at least one
       prior valid non-repeat recall, add ``log(1 - p_stop_pre_drift)``
       using the current c_ret. Stopping is NOT consulted on repeats
       (Tier-1 skips stopping + scoring on repeats; only prev_idx advances).
    2. If this row is valid AND NOT a repeat AND has_prior, drift c_ret
       toward e_{prev_idx+1} (C&Z 2025 Eq 3; identity M^FC_pre target).
    3. If this row is valid AND NOT a repeat, compute log_softmax of
       k·M^IC^T@c_ret (masking already_mask) and add log_p[sp-1].
    4. On valid non-repeat rows: mark sp-1 as recalled; reset c_ret to
       c_item_traj[sp] (reactivation; C&Z 2025 Fig 1b); advance prev_idx.
       On valid REPEAT rows: leave already_mask, c_ret, and log_l
       unchanged; advance prev_idx to this sp's idx (so the next valid
       non-repeat's drift targets the repeat's sp, matching Tier-1
       ``prev_sp = sp; continue``).
    """
    c_ret, already_mask, log_l, prev_idx = state
    sp, valid = step_inputs
    # sp is 1-based; subtract 1 for zero-based internal indexing.
    idx = xp.clip(sp - 1, 0, W - 1)
    is_repeat = already_mask[idx]
    # A "countable" step is valid AND not a repeat: only countable steps
    # contribute stopping + score terms and update c_ret / already_mask.
    countable = valid & ~is_repeat

    # (1) Stopping-rule contribution on pre-drift c_ret.
    _, log_1m_pstop, _ = stopping_log_terms(
        m_ic, c_ret, already_mask, hp, xp,
    )
    has_prior = prev_idx >= 0
    stop_contrib = xp.where(has_prior & countable, log_1m_pstop, 0.0)
    log_l = log_l + stop_contrib

    # (2) Apply beta_rec drift on countable non-first recalls.
    d = c_ret.shape[0]
    prev_idx_safe = xp.clip(prev_idx + 1, 0, d - 1)
    # Drift target is e_{prev_idx+1}. dot(c_ret, e_i) = c_ret[i].
    dot_r = c_ret[prev_idx_safe]
    rho_r = norm_preserving_rho(hp.beta_rec, dot_r, xp)
    c_ret_drifted_full = rho_r * c_ret
    c_ret_drifted_full = at_add(
        c_ret_drifted_full, prev_idx_safe, hp.beta_rec, xp,
    )
    apply_drift = has_prior & countable
    c_ret_drifted = xp.where(apply_drift, c_ret_drifted_full, c_ret)

    # (3) Score against list (on countable rows only).
    a = activation(m_ic, c_ret_drifted, xp)
    keep_mask = ~already_mask
    log_p_all = log_softmax_masked(hp.k * a, keep_mask, xp)
    score_contrib = xp.where(countable, log_p_all[idx], 0.0)
    log_l = log_l + score_contrib

    # (4) Update state.
    # - already_mask: mark sp-1 on countable rows; repeats/padding leave it.
    # - c_ret: reset to c_item_traj[sp] on countable; repeats/padding leave
    #   c_ret at c_ret_drifted (which, when countable=False, is identical
    #   to the original c_ret since apply_drift=False → no drift applied).
    # - prev_idx: advance to idx whenever valid (repeat or not), matching
    #   Tier-1's ``prev_sp = sp; continue`` on repeats.
    sp_safe = xp.clip(sp, 0, W)
    new_already = xp.where(
        countable,
        at_set(already_mask, idx, True, xp),
        already_mask,
    )
    new_c_ret = xp.where(
        countable,
        c_item_traj[sp_safe],
        c_ret_drifted,
    )
    new_prev_idx = xp.where(valid, idx, prev_idx)

    return (new_c_ret, new_already, log_l, new_prev_idx)


def run_recalls(
    hp: CoreHyperparams,
    c_item_traj,
    m_ic,
    recall_sps,
    recall_mask,
    W: int,
    xp,
    dtype,
):
    """Drive the recall-step scan; returns (log_l, c_ret_final, already_final, prev_idx_final)."""
    c_ret_init = c_item_traj[W]  # end-of-list cue
    already_init = xp.zeros(W, dtype=bool)
    log_l_init = xp.asarray(0.0, dtype=dtype)
    prev_idx_init = xp.asarray(-1, dtype=xp.int32) if _is_jax(xp) else -1
    state = (c_ret_init, already_init, log_l_init, prev_idx_init)

    if _is_jax(xp):
        import jax

        def scan_body(state, inp):
            new_state = recall_step(
                state, inp, hp, W, c_item_traj, m_ic, xp, dtype,
            )
            return new_state, None

        final_state, _ = jax.lax.scan(
            scan_body, state, (recall_sps, recall_mask),
        )
        return final_state

    # numpy path
    R = len(recall_sps)
    for r in range(R):
        state = recall_step(
            state,
            (int(recall_sps[r]), bool(recall_mask[r])),
            hp, W, c_item_traj, m_ic, xp, dtype,
        )
    return state


# --- top-level per-list likelihood ---------------------------------------


def compute_list_log_likelihood(
    hp: CoreHyperparams,
    cat_indices,
    recall_sps,
    recall_mask,
    W: int,
    K: int,
    xp,
    dtype,
):
    """End-to-end per-list log-likelihood.

    Semantics (must match ``ms_tcm.likelihood.list_log_likelihood`` exactly):

    - Encoding runs for W steps producing c_item_traj (W+1, d) and M^IC.
    - First recall: softmax over k · M^IC^T @ c_item_end.
    - Each subsequent recall contributes log(1 - p_stop) on the pre-drift
      c_ret, then drift c_ret, then log(softmax(k · a)) at the observed sp.
    - After all recalls: log(p_stop) at termination.
    - Free-recall-only stopping terms are ACTIVE when paradigm=='free_recall';
      the hyperparameter object does not carry paradigm, so stopping is
      always included here. Callers in cued-recall drop stopping contributions
      by setting ε_d → large (p_stop → 0) — but in practice MS-TCM on the
      FRFR-category dataset is always free-recall. If cued-recall support
      is needed, the caller passes a flag to skip stopping; the current
      FRFR-only API does not need that path.
    """
    c_item_traj, m_ic = run_encoding(hp, cat_indices, W, K, xp, dtype)

    c_ret_final, already_final, log_l_after_scan, prev_idx_final = run_recalls(
        hp, c_item_traj, m_ic, recall_sps, recall_mask, W, xp, dtype,
    )

    # Terminal stopping contribution: log(p_stop) after last valid recall.
    _, _, log_pstop = stopping_log_terms(
        m_ic, c_ret_final, already_final, hp, xp,
    )
    log_l_final = log_l_after_scan + xp.where(
        prev_idx_final >= 0, log_pstop, 0.0,
    )
    return log_l_final


# --- numpy convenience wrapper -------------------------------------------


def compute_list_log_likelihood_numpy(
    params,
    cat_indices,
    recall_sps,
    recall_mask,
    W: int,
    K: int,
) -> float:
    """Numpy-dtype entry point (Tier-1 and oracle use this)."""
    import numpy as np

    hp = CoreHyperparams.from_model_parameters(params)
    ll = compute_list_log_likelihood(
        hp,
        np.asarray(cat_indices, dtype=np.int64),
        np.asarray(recall_sps, dtype=np.int64),
        np.asarray(recall_mask, dtype=bool),
        W=W, K=K, xp=np, dtype=np.float64,
    )
    return float(ll)
