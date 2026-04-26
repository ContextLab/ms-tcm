"""Multi-Storyline TCM (MS-TCM): C&Z 2025 hierarchical CMR + per-storyline
context tracking + λ storyline-return reinstatement + τ initiation mixture.

Extends the verified C&Z 2025 implementation in ``_likelihood_core.py``
with two new mechanisms targeted at categorically-organized free recall:

1. **Per-storyline contexts (v1 §3.1, §3.2)**: track c^story_s for each
   storyline s ∈ {1, ..., K}, drifting only when items from storyline s
   are encoded. Inactive storylines remain frozen.

2. **λ storyline-return reinstatement (v6 §2.4 Eq 6)**: when a storyline
   resumes after interruption, reinstate the cached storyline context
   from M^SC.

3. **τ storyline-initiation mixture (NEW)**: at recall onset, with
   probability τ the first recall is initiated via storyline retrieval
   (select ŝ ∝ softmax(k · M^SC · c^list_end), reactivate item context
   to c^story_ŝ, then sample the first recall under standard C&Z
   dynamics); with probability (1-τ) the first recall is initiated by
   the standard recency cue (c^item_end). After the first recall, the
   model proceeds with C&Z's standard retrieval (β_rec drift + stopping
   rule + e_start fallback when item-level fails).

   This explains pFR=1 primacy-initiation observed in FRFR-category
   (and other categorically-structured free-recall tasks): when τ is
   substantial, the first recall is biased toward early items of the
   first-encoded storyline.

Reductions:
- K=1, τ=0  ⇒ exactly C&Z 2025 (single-storyline, standard initiation)
- K=1, τ=0, λ ignored ⇒ exactly C&Z 2025
- All-items-from-one-storyline ⇒ MS-TCM degenerates to C&Z (storyline
  context evolves like list-level context; no return reinstatements
  occur because there's only one storyline).

Code reuse: ``_compute_ll_per_T`` from ``_likelihood_core`` is reused
verbatim for both routes (recency and storyline-init) — they differ only
in the initial c_ret seed for the C&Z marginalized LL machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

from ms_tcm._likelihood_core import (
    _compute_ll_per_T,
    run_encoding as _run_encoding_cz,  # not used here; kept for reference
)
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
class MSCoreHyperparams:
    """Scalar MS-TCM parameters.

    Inherited from C&Z (Table 1):
        beta_enc, beta_list, beta_rec, beta_rein, gamma_fc, k, epsilon_d.

    New in MS-TCM:
        lambda_reinstate ∈ [0, 1]: storyline-return reinstatement strength
            at encoding (v6 §2.4 Eq 6). When a storyline resumes after
            interruption, c^story_s ← λ · c̃^story_s + (1-λ) · c^story_s_prev,
            where c̃ is the cached context from M^SC at the storyline's
            last departure.
        tau_init ∈ [0, 1]: probability of storyline-initiation route at
            recall onset. tau=0 reduces to C&Z; tau=1 always initiates
            via storyline retrieval.
    """

    beta_enc: float
    beta_list: float
    beta_rec: float
    beta_rein: float  # reserved for FFR; unused in single-list FR
    gamma_fc: float
    k: float
    epsilon_d: float
    lambda_reinstate: float
    tau_init: float

    @classmethod
    def from_model_parameters(cls, p) -> "MSCoreHyperparams":
        return cls(
            beta_enc=float(p.beta_enc),
            beta_list=float(p.beta_list),
            beta_rec=float(p.beta_rec),
            beta_rein=float(p.beta_rein),
            gamma_fc=float(p.gamma_fc),
            k=float(p.k),
            epsilon_d=float(p.epsilon_d),
            lambda_reinstate=float(p.lambda_reinstate),
            tau_init=float(p.tau_init),
        )


# --- encoding (per-storyline + λ reinstatement + M^SC accumulation) ------


def run_encoding_mstcm(
    hp: MSCoreHyperparams,
    W: int,
    K: int,
    cat_indices,    # (W,) int: storyline index 0..K-1 per encoding step
    xp,
    dtype,
):
    """Run W-step MS-TCM encoding; return per-storyline contexts + matrices.

    Returns:
        c_item_traj  : (W+1, d)         standard item-level context trajectory
        c_list_end   : (d,)             list-level context at end of encoding
        c_story_end  : (K, d)           per-storyline context at end of encoding
        m_fc_exp     : (d, W)           experimental context→item matrix (Eq 2a)
        m_cf_exp     : (W, d)           experimental item→context matrix (Eq 2b)
        m_sc         : (K, d)           cached storyline contexts at switches/returns

    Storyline storage convention: M^SC[s, :] holds the cached c^story_s
    context written when storyline s was last encoded before a switch
    (or, after a return, the post-reinstatement state). Used at retrieval
    for storyline-init and as the M^lists analog of C&Z Eq 12.

    Math (per encoding step t with item-storyline s = cat_indices[t]):
      1. (Eq 2a, 2b updates with PRE-drift c_item)
      2. (Eq 1 item drift)
      3. (Eq 8 list drift)
      4. **per-storyline drift**:
         - if s active for the first time, c^story_s drifts at β_list
           toward c^IN_t (initialize from e_start for first appearance)
         - if s was just resumed (storyline switch where s seen before):
           cache c^story_prev to M^SC[prev_s], then c^story_s ← λ · M^SC[s]
           + (1-λ) · c^story_s_prev (v6 Eq 6); then drift c^story_s by β_list
         - inactive storylines stay frozen

    Identity M^FC_pre: c^IN_t = e_{t+1} (basis vector).
    """
    d = W + 1
    e_start = xp.zeros(d, dtype=dtype)
    e_start = at_set(e_start, 0, 1.0, xp)

    if _is_jax(xp):
        return _run_encoding_mstcm_jax(
            hp, W, K, cat_indices, xp, dtype, e_start,
        )

    # numpy path
    c_item = e_start.copy()
    c_list = e_start.copy()
    # Per-storyline contexts (K, d). Each storyline starts from e_start.
    c_story = xp.tile(e_start[None, :], (K, 1)).copy()
    m_fc_exp = xp.zeros((d, W), dtype=dtype)
    m_cf_exp = xp.zeros((W, d), dtype=dtype)
    m_sc = xp.zeros((K, d), dtype=dtype)
    c_item_traj = xp.zeros((W + 1, d), dtype=dtype)
    c_item_traj[0] = c_item

    # Boolean array: has storyline s been seen yet?
    seen = xp.zeros(K, dtype=bool)
    prev_s = -1  # storyline index of previous encoded item; -1 at start

    for t in range(W):
        s = int(cat_indices[t])

        # Eq 2a, 2b: pre-drift c_item.
        m_fc_exp[:, t] = m_fc_exp[:, t] + c_item
        m_cf_exp[t, :] = m_cf_exp[t, :] + c_item

        # Eq 1: c_item drift.
        c_in = e_start * 0.0  # basis vector e_{t+1}
        c_in = at_set(c_in, t + 1, 1.0, xp)
        c_item = drift_c_ret(c_item, c_in, hp.beta_enc, xp)

        # Eq 8: c_list drift.
        c_list = drift_c_ret(c_list, c_in, hp.beta_list, xp)

        # --- Per-storyline drift + λ reinstatement ---
        is_switch = (prev_s >= 0) and (prev_s != s)
        is_return = is_switch and bool(seen[s])

        if is_switch:
            # Cache c^story_prev_s to M^SC[prev_s].
            m_sc[prev_s, :] = c_story[prev_s, :]

        if is_return:
            # v6 Eq 6: blend cached and current contexts by λ.
            cached = m_sc[s, :]
            current = c_story[s, :]
            c_story[s, :] = (hp.lambda_reinstate * cached
                             + (1.0 - hp.lambda_reinstate) * current)
            # Renormalize to unit norm (interpolated vectors aren't unit).
            n = xp.sqrt(xp.sum(c_story[s, :] ** 2))
            c_story[s, :] = c_story[s, :] / xp.maximum(n, 1e-12)

        # Drift the active storyline by β_list (using c_list's drift rate
        # since storyline drift inherits the list-level rate per v6 §2.1).
        c_story_active = c_story[s, :]
        c_story[s, :] = drift_c_ret(c_story_active, c_in, hp.beta_list, xp)

        seen = at_set(seen, s, True, xp)
        prev_s = s
        c_item_traj[t + 1] = c_item

    # At end of encoding, ensure the LAST active storyline's cache is
    # also written to M^SC (so M^SC reflects all storylines' final
    # encoding contexts). This is needed for the storyline-init route
    # at retrieval.
    if prev_s >= 0:
        m_sc[prev_s, :] = c_story[prev_s, :]

    return c_item_traj, c_list, c_story, m_fc_exp, m_cf_exp, m_sc


def _run_encoding_mstcm_jax(hp, W, K, cat_indices, xp, dtype, e_start):
    """JAX scan-based encoding for MS-TCM (parallel structure to numpy path)."""
    import jax

    d = W + 1

    def step(carry, inputs):
        c_item, c_list, c_story, m_fc_exp, m_cf_exp, m_sc, seen, prev_s = carry
        t, s = inputs

        # Eq 2a, 2b: pre-drift c_item.
        m_fc_exp = at_set(
            m_fc_exp, (slice(None), t), m_fc_exp[:, t] + c_item, xp,
        )
        m_cf_exp = at_set(
            m_cf_exp, (t, slice(None)), m_cf_exp[t, :] + c_item, xp,
        )

        # c^IN = e_{t+1}.
        c_in = xp.zeros(d, dtype=dtype)
        c_in = at_set(c_in, t + 1, 1.0, xp)

        # Eq 1: c_item drift.
        c_item = drift_c_ret(c_item, c_in, hp.beta_enc, xp)
        # Eq 8: c_list drift.
        c_list = drift_c_ret(c_list, c_in, hp.beta_list, xp)

        # Boundary detection.
        is_switch = (prev_s >= 0) & (s != prev_s)
        is_return = is_switch & seen[s]

        # Cache c^story_prev_s on switch.
        new_m_sc = xp.where(
            is_switch,
            at_set(m_sc, prev_s, c_story[prev_s, :], xp),
            m_sc,
        )

        # On return: λ reinstatement.
        cached = new_m_sc[s, :]
        current = c_story[s, :]
        blended = (hp.lambda_reinstate * cached
                   + (1.0 - hp.lambda_reinstate) * current)
        n = xp.sqrt(xp.sum(blended * blended))
        blended_normalized = blended / xp.maximum(n, 1e-12)
        c_story_after_return = xp.where(
            is_return, blended_normalized, current,
        )
        c_story = at_set(c_story, s, c_story_after_return, xp)

        # Per-storyline drift on active storyline.
        c_story_active = c_story[s, :]
        c_story_drifted = drift_c_ret(
            c_story_active, c_in, hp.beta_list, xp,
        )
        c_story = at_set(c_story, s, c_story_drifted, xp)

        seen = at_set(seen, s, True, xp)
        return (
            (c_item, c_list, c_story, m_fc_exp, m_cf_exp, new_m_sc, seen, s),
            c_item,
        )

    # Initialize.
    c_item0 = e_start
    c_list0 = e_start
    c_story0 = xp.tile(e_start[None, :], (K, 1))
    m_fc_exp0 = xp.zeros((d, W), dtype=dtype)
    m_cf_exp0 = xp.zeros((W, d), dtype=dtype)
    m_sc0 = xp.zeros((K, d), dtype=dtype)
    seen0 = xp.zeros(K, dtype=bool)
    prev_s0 = xp.asarray(-1, dtype=xp.int32)

    init_carry = (
        c_item0, c_list0, c_story0, m_fc_exp0, m_cf_exp0,
        m_sc0, seen0, prev_s0,
    )
    t_arr = xp.arange(W, dtype=xp.int32)
    cat_indices_jax = xp.asarray(cat_indices, dtype=xp.int32)

    final_carry, c_item_steps = jax.lax.scan(
        step, init_carry, (t_arr, cat_indices_jax),
    )
    (c_item_f, c_list_f, c_story_f, m_fc_exp_f, m_cf_exp_f,
     m_sc_f, seen_f, prev_s_f) = final_carry

    # Final M^SC update (last active storyline).
    m_sc_final = xp.where(
        prev_s_f >= 0,
        at_set(m_sc_f, prev_s_f, c_story_f[prev_s_f, :], xp),
        m_sc_f,
    )

    c_item_traj = xp.concatenate([e_start[None, :], c_item_steps], axis=0)
    return c_item_traj, c_list_f, c_story_f, m_fc_exp_f, m_cf_exp_f, m_sc_final


# --- per-list log-likelihood --------------------------------------------


def compute_list_log_likelihood_mstcm(
    hp: MSCoreHyperparams,
    W: int,
    K: int,
    cat_indices,   # (W,) int: storyline label per encoding step
    recall_sps,    # (R,) int: 1-based serial positions
    recall_mask,   # (R,) bool: True for valid recall slots
    xp,
    dtype,
):
    """End-to-end per-list LL under MS-TCM (mixture over initiation routes).

    Computes:
        LL = log[
              (1 - τ) · LL_recency_path
            + τ · Σ_ŝ P(ŝ) · LL_storyline_path(ŝ)
        ]
    where P(ŝ) = softmax(k · M^SC · c^list_end), and each LL_path computes
    the C&Z marginalized LL (over the phase-transition latent T) from a
    different initial c_ret seed:
        LL_recency_path: c_ret_init = c_item_end (standard C&Z)
        LL_storyline_path(ŝ): c_ret_init = c^story_ŝ

    The phase-2 fallback target (when item-level retrieval fails) is
    e_start in both routes — matching C&Z's free-recall reduction.
    """
    d = W + 1

    # 1. Encode.
    c_item_traj, c_list_end, c_story_end, m_fc_exp, m_cf_exp, m_sc = (
        run_encoding_mstcm(hp, W, K, cat_indices, xp, dtype)
    )

    # 2. Compute storyline-selection log-probabilities.
    # a_s = M^SC[s] · c^list_end (cosine-style readout via dot).
    storyline_acts = m_sc @ c_list_end  # (K,)
    log_p_s = log_softmax_masked(
        hp.k * storyline_acts,
        xp.ones(K, dtype=bool),  # all storylines available
        xp,
    )  # (K,)

    # 3. Phase-2 fallback always targets e_start.
    e_start = xp.zeros(d, dtype=dtype)
    e_start = at_set(e_start, 0, 1.0, xp)

    # 4. Compute LL per route.
    # For each candidate seed (recency + K storylines), call C&Z's
    # `_compute_ll_per_T` with the seed as initial c_ret_phase1 and
    # marginalize the result via logsumexp.
    R_total = recall_sps.shape[0]

    def _ll_for_seed(c_ret_init):
        """C&Z marginalized LL using the given initial c_ret seed."""
        # We need the same forward-phase machinery C&Z uses; reuse its
        # internal _compute_ll_per_T by first computing the phase-1
        # forward (cum_ll_phase1, c_ret_traj_phase1, mask_traj).
        # Manual scan, mirroring C&Z's _forward_phase logic.
        cum_ll_p1, c_ret_traj_p1, mask_traj = _forward_phase_seeded(
            c_ret_init, m_fc_exp, m_cf_exp, hp, W, R_total,
            recall_sps, recall_mask, xp, dtype,
        )
        ll_per_T = _compute_ll_per_T(
            R_total=R_total, W=W, d=d,
            c_ret_traj_phase1=c_ret_traj_p1,
            cum_ll_phase1=cum_ll_p1,
            mask_traj=mask_traj,
            recall_sps=recall_sps, recall_mask=recall_mask,
            m_fc_exp=m_fc_exp, m_cf_exp=m_cf_exp,
            hp=_hp_to_cz_compat(hp), xp=xp, dtype=dtype,
            c_ret_init_phase2=e_start,
        )
        # logsumexp over T.
        m = xp.max(ll_per_T)
        return m + xp.log(xp.sum(xp.exp(ll_per_T - m)))

    # Recency route LL.
    ll_recency = _ll_for_seed(c_item_traj[W])

    # Storyline route: weighted sum over storylines.
    if _is_jax(xp):
        import jax
        ll_storylines = jax.vmap(_ll_for_seed)(c_story_end)  # (K,)
        ll_storyline_route = jax.scipy.special.logsumexp(log_p_s + ll_storylines)
    else:
        # numpy path: explicit loop.
        ll_storylines = xp.array(
            [float(_ll_for_seed(c_story_end[s, :])) for s in range(K)],
            dtype=dtype,
        )
        combined = log_p_s + ll_storylines
        m = combined.max()
        ll_storyline_route = m + xp.log(xp.sum(xp.exp(combined - m)))

    # Mix the two routes.
    log_tau = xp.log(xp.maximum(hp.tau_init, 1e-300))
    log_1m_tau = xp.log(xp.maximum(1.0 - hp.tau_init, 1e-300))
    a = log_1m_tau + ll_recency
    b = log_tau + ll_storyline_route
    m = xp.maximum(a, b)
    return m + xp.log(xp.exp(a - m) + xp.exp(b - m))


# --- helper: phase-1 forward seeded with arbitrary c_ret_init ------------


def _forward_phase_seeded(
    c_ret_init, m_fc_exp, m_cf_exp, hp, W, R_total,
    recall_sps, recall_mask, xp, dtype,
):
    """Forward pass through R_total slots with an arbitrary c_ret seed.

    Mirrors the phase-1 forward in ``_likelihood_core._forward_phase`` —
    factored here so MS-TCM can run it with different seeds (recency vs
    storyline initiation). Returns (cum_ll, c_ret_traj, mask_traj) all of
    leading length R_total + 1.

    Repeats are treated as noise (no LL contribution, no state update).
    """
    d = W + 1
    if _is_jax(xp):
        import jax

        def step(carry, slot):
            c_ret, cum, already_mask = carry
            sp, valid = slot
            idx = xp.clip(sp - 1, 0, W - 1)
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

            c_IN = c_IN_rec_of(idx, m_fc_exp, hp.gamma_fc, d, xp, dtype)
            c_ret_drifted = drift_c_ret(c_ret, c_IN, hp.beta_rec, xp)
            new_c_ret = xp.where(countable, c_ret_drifted, c_ret)
            new_mask = xp.where(
                countable, at_set(already_mask, idx, True, xp), already_mask,
            )
            return (new_c_ret, new_cum, new_mask), (new_c_ret, new_cum, new_mask)

        init = (
            c_ret_init,
            xp.asarray(0.0, dtype=dtype),
            xp.zeros(W, dtype=bool),
        )
        _, (c_ret_1toR, cum_1toR, mask_1toR) = jax.lax.scan(
            step, init, (recall_sps, recall_mask),
        )
        c_ret_traj = xp.concatenate([c_ret_init[None, :], c_ret_1toR], axis=0)
        cum_ll = xp.concatenate(
            [xp.zeros((1,), dtype=dtype), cum_1toR], axis=0,
        )
        mask_traj = xp.concatenate(
            [xp.zeros((1, W), dtype=bool), mask_1toR], axis=0,
        )
        return cum_ll, c_ret_traj, mask_traj

    # numpy
    c_ret = xp.array(c_ret_init)
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


def _hp_to_cz_compat(hp: MSCoreHyperparams):
    """Adapt MSCoreHyperparams to the CZ ``CoreHyperparams`` shape that
    ``_compute_ll_per_T`` expects (it accesses .epsilon_d, .k, .beta_rec,
    .gamma_fc fields).

    The MS-TCM-only fields (lambda_reinstate, tau_init) are not read by
    ``_compute_ll_per_T``, so a lightweight compatible dataclass works.
    """
    from ms_tcm._likelihood_core import CoreHyperparams
    return CoreHyperparams(
        beta_enc=hp.beta_enc,
        beta_list=hp.beta_list,
        beta_rec=hp.beta_rec,
        beta_rein=hp.beta_rein,
        gamma_fc=hp.gamma_fc,
        k=hp.k,
        epsilon_d=hp.epsilon_d,
    )


# --- numpy entry point ---------------------------------------------------


def compute_list_log_likelihood_mstcm_numpy(
    params,
    cat_indices,
    recall_sps,
    recall_mask,
    W: int,
    K: int,
) -> float:
    """Numpy entry point: build hp from ModelParameters, dispatch via xp=numpy."""
    import numpy as np

    hp = MSCoreHyperparams.from_model_parameters(params)
    ll = compute_list_log_likelihood_mstcm(
        hp, W=W, K=K,
        cat_indices=np.asarray(cat_indices, dtype=np.int64),
        recall_sps=np.asarray(recall_sps, dtype=np.int64),
        recall_mask=np.asarray(recall_mask, dtype=bool),
        xp=np, dtype=np.float64,
    )
    return float(ll)


# --- simulator (forward sampler) -----------------------------------------


def simulate_recalls_mstcm(
    params,
    W: int,
    K: int,
    cat_indices,
    rng,
    *,
    max_recalls: int | None = None,
) -> list[int]:
    """Sample a recall sequence from MS-TCM (mixture initiation + C&Z).

    Implements the generative process literally:
      1. Encode → produce c_item_traj, c_story_end, M^SC, M^FC_exp, M^CF_exp.
      2. Initiation:
         - Draw u ~ Uniform[0,1].
         - If u < τ: draw ŝ ∝ softmax(k · M^SC · c^list_end); set
           c_ret = c^story_ŝ (or, equivalently, sample first recall under
           c_ret = c^story_ŝ).
         - Else: c_ret = c_item_end.
      3. Standard C&Z retrieval from there.
    """
    import numpy as np

    hp = MSCoreHyperparams.from_model_parameters(params)
    d = W + 1

    c_item_traj, c_list_end, c_story_end, m_fc_exp, m_cf_exp, m_sc = (
        run_encoding_mstcm(
            hp, W, K, np.asarray(cat_indices, dtype=np.int64),
            np, np.float64,
        )
    )

    # Initiation route.
    if rng.random() < hp.tau_init:
        # Storyline-init route.
        storyline_acts = m_sc @ c_list_end
        # softmax(k · a) over storylines.
        scaled = hp.k * storyline_acts
        m = scaled.max()
        p = np.exp(scaled - m)
        p = p / p.sum()
        s_hat = int(rng.choice(K, p=p))
        c_ret = c_story_end[s_hat].copy()
    else:
        c_ret = c_item_traj[W].copy()

    # Standard C&Z retrieval from here on.
    e_start = np.zeros(d, dtype=np.float64)
    e_start[0] = 1.0
    phase = 1
    already_mask = np.zeros(W, dtype=bool)
    recalls: list[int] = []
    cap = max_recalls if max_recalls is not None else 3 * W

    while len(recalls) < cap:
        p_stop = float(compute_p_stop(
            m_cf_exp, c_ret, already_mask, hp.epsilon_d, np,
        ))
        if rng.random() < p_stop:
            if phase == 1:
                c_ret = e_start.copy()
                phase = 2
                continue
            else:
                break
        a = activation(m_cf_exp, c_ret, np)
        log_p = log_softmax_masked(hp.k * a, ~already_mask, np)
        p = np.exp(log_p - log_p.max())
        p = np.where(already_mask, 0.0, p)
        p_sum = p.sum()
        if p_sum <= 0:
            break
        p = p / p_sum
        idx = int(rng.choice(W, p=p))
        recalls.append(idx + 1)
        already_mask[idx] = True
        c_IN = c_IN_rec_of(idx, m_fc_exp, hp.gamma_fc, d, np, np.float64)
        c_ret = drift_c_ret(c_ret, c_IN, hp.beta_rec, np)

    return recalls
