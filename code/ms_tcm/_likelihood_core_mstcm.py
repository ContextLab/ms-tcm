"""MS-TCM MS-TCM: strict hierarchical recall with separate global +
per-storyline contexts.

Extends pre-consolidation MS-TCM (`_likelihood_core_mstcm.py`) with:
1. **Separate global context** that drifts at every encoding step (rate
   β_enc_global), distinct from per-storyline contexts.
2. **Strict hierarchical retrieval**: route β samples a storyline from the
   global context, then recalls items from that storyline's
   per-storyline matrix until that storyline stops, then re-runs global
   selection (with the immediately-preceding storyline excluded — see
   architecture log OQ2).
3. **Per-storyline associative matrices** (M^FC_s, M^CF_s) — separate
   from the global matrices (M^FC_G, M^CF_G).

See `notes/mstcm_architecture_log.md` for the full design rationale,
including resolved OQ1-OQ3:
- OQ1=a: route α uses M^CF_G only (pure-global recall)
- OQ2: track only the IMMEDIATELY-preceding storyline (single-int mask)
- OQ3=ii: c_ret_s = e_start at storyline start (option iii deferred)

Reductions (sanity properties):
- K=1: per-storyline = global; reduces to pre-consolidation MS-TCM.
- K=1 + τ=0: pure-global retrieval, no storyline routing; reduces to C&Z
  flat retrieval (no hierarchical fallback).

Equation reference (subscripts):
    Per-storyline (one per category s):
        c^item_s drifts at β_enc on storyline-s items, frozen otherwise
        c^story_s drifts at β_list on storyline-s items, frozen otherwise
        M^FC_s, M^CF_s built from storyline-s items only
    Global (one set across the list):
        c^global drifts at β_enc_global on EVERY item
        c^list_G drifts at β_list on EVERY item
        M^FC_G, M^CF_G built from ALL items
        M^lists_G[s] = c^story_s_end (cached storyline contexts at end)

Likelihood computation: top-level mixture over routes α/β (probability
1−τ vs τ). Within route β, the observed cross-category transitions
deterministically reveal the storyline-visit boundaries (no
marginalization needed there). Each within-storyline visit contributes
a simple cumulative LL plus a final stopping-rule term to fire the
boundary.
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


# --- hyperparameter bundle ----------------------------------------------


@dataclass(frozen=True)
class MSCoreHyperparams:
    """Scalar MS-TCM MS-TCM parameters.

    Inherited from C&Z + pre-consolidation MS-TCM:
        beta_enc, beta_list, beta_rec, beta_rein, gamma_fc, k, epsilon_d,
        lambda_reinstate, tau_init.

    NEW in MS-TCM:
        beta_enc_global ∈ (0, 1): global cross-storyline encoding drift.
    """

    beta_enc: float
    beta_enc_global: float
    beta_list: float
    beta_rec: float
    beta_rein: float
    gamma_fc: float
    k: float
    epsilon_d: float
    lambda_reinstate: float
    tau_init: float

    @classmethod
    def from_model_parameters(cls, p) -> "MSCoreHyperparams":
        return cls(
            beta_enc=float(p.beta_enc),
            beta_enc_global=float(p.beta_enc_global),
            beta_list=float(p.beta_list),
            beta_rec=float(p.beta_rec),
            beta_rein=float(p.beta_rein),
            gamma_fc=float(p.gamma_fc),
            k=float(p.k),
            epsilon_d=float(p.epsilon_d),
            lambda_reinstate=float(p.lambda_reinstate),
            tau_init=float(p.tau_init),
        )


# --- encoding ------------------------------------------------------------


def run_encoding_mstcm(
    hp: MSCoreHyperparams,
    W: int,
    K: int,
    cat_indices,    # (W,) int: storyline label per encoding step
    xp,
    dtype,
):
    """Run W-step MS-TCM encoding.

    Returns:
        c_item_traj_g  : (W+1, d) global item-level context trajectory
        c_item_per_s   : (K, d)   per-storyline item-level contexts at end
        c_story_per_s  : (K, d)   per-storyline list-level contexts at end
        c_list_g_end   : (d,)     global list-level context at end
        m_fc_g         : (d, W)   global M^FC_exp
        m_cf_g         : (W, d)   global M^CF_exp
        m_fc_s         : (K, d, W)   per-storyline M^FC_exp (one per storyline)
        m_cf_s         : (K, W, d)   per-storyline M^CF_exp
        m_sc           : (K, d)   M^lists_G[s] = cached storyline list-context

    Per-step semantics (item from storyline s = cat_indices[t]):

        c^IN_t = e_{t+1}  (basis vector under identity M^FC_pre)

        # Global branch — updates EVERY step:
        m_fc_g[:, t]   += c^global  (pre-drift)
        m_cf_g[t, :]   += c^global
        c^global       ← drift(c^global, c^IN_t, β_enc_global)
        c^list_g       ← drift(c^list_g, c^IN_t, β_list)

        # Per-storyline branch — updates ONLY storyline-s matrices:
        m_fc_s[s, :, t] += c^item_s                  (pre-drift, per s)
        m_cf_s[s, t, :] += c^item_s
        c^item_s        ← drift(c^item_s, c^IN_t, β_enc)
        c^story_s       ← drift(c^story_s, c^IN_t, β_list)

        # All storylines OTHER THAN s remain frozen this step.

        # λ storyline-return reinstatement (v6 §2.4): if storyline s has
        # been seen before AND the previous step was a different storyline
        # (i.e., this is a return), blend cached + current storyline contexts:
        #     c^story_s ← λ · M^SC[s] + (1 − λ) · c^story_s
        # before the storyline-s drift.

    NOTE: This function is the largest piece of new math in MS-TCM.
    The numpy and JAX paths must match bit-for-bit; both will exercise
    the same xp-dispatched primitives.
    """
    d = W + 1
    e_start = xp.zeros(d, dtype=dtype)
    e_start = at_set(e_start, 0, 1.0, xp)

    if _is_jax(xp):
        return _run_encoding_mstcm_jax(
            hp, W, K, cat_indices, xp, dtype, e_start,
        )

    # --- numpy path ---
    c_global = e_start.copy()
    c_list_g = e_start.copy()
    # Per-storyline contexts: K rows, each starts at e_start.
    c_item_per_s = xp.tile(e_start[None, :], (K, 1)).copy()
    c_story_per_s = xp.tile(e_start[None, :], (K, 1)).copy()

    m_fc_g = xp.zeros((d, W), dtype=dtype)
    m_cf_g = xp.zeros((W, d), dtype=dtype)
    m_fc_s = xp.zeros((K, d, W), dtype=dtype)
    m_cf_s = xp.zeros((K, W, d), dtype=dtype)
    m_sc = xp.zeros((K, d), dtype=dtype)

    c_item_traj_g = xp.zeros((W + 1, d), dtype=dtype)
    c_item_traj_g[0] = c_global

    seen = xp.zeros(K, dtype=bool)
    prev_s = -1

    for t in range(W):
        s = int(cat_indices[t])
        is_switch = (prev_s >= 0) and (prev_s != s)
        is_return = is_switch and bool(seen[s])

        # On switch: cache the OUTGOING storyline's c^story_prev_s into M^SC.
        if is_switch:
            m_sc[prev_s, :] = c_story_per_s[prev_s, :]

        # Decision 1A: λ reinstatement BEFORE per-storyline drift on returns.
        if is_return:
            cached = m_sc[s, :].copy()
            current = c_story_per_s[s, :].copy()
            blended = hp.lambda_reinstate * cached + (1.0 - hp.lambda_reinstate) * current
            n = float(xp.sqrt(xp.sum(blended * blended)))
            if n > 1e-12:
                blended = blended / n
            c_story_per_s[s, :] = blended

        # c^IN_t = e_{t+1} (basis vector under identity M^FC_pre).
        c_in = xp.zeros(d, dtype=dtype)
        c_in[t + 1] = 1.0

        # --- Associative matrix updates: PRE-drift contexts ---
        # Global branch — always.
        m_fc_g[:, t] = m_fc_g[:, t] + c_global
        m_cf_g[t, :] = m_cf_g[t, :] + c_global
        # Per-storyline branch — storyline s only.
        m_fc_s[s, :, t] = m_fc_s[s, :, t] + c_item_per_s[s, :]
        m_cf_s[s, t, :] = m_cf_s[s, t, :] + c_item_per_s[s, :]

        # --- Drifts ---
        # Global drifts every step.
        c_global = drift_c_ret(c_global, c_in, hp.beta_enc_global, xp)
        c_list_g = drift_c_ret(c_list_g, c_in, hp.beta_list, xp)
        # Storyline s only — others frozen.
        c_item_per_s[s, :] = drift_c_ret(
            c_item_per_s[s, :], c_in, hp.beta_enc, xp,
        )
        c_story_per_s[s, :] = drift_c_ret(
            c_story_per_s[s, :], c_in, hp.beta_list, xp,
        )

        seen[s] = True
        prev_s = s
        c_item_traj_g[t + 1] = c_global

    # End-of-encoding: cache the last active storyline's context.
    if prev_s >= 0:
        m_sc[prev_s, :] = c_story_per_s[prev_s, :]

    return (
        c_item_traj_g, c_item_per_s, c_story_per_s, c_list_g,
        m_fc_g, m_cf_g, m_fc_s, m_cf_s, m_sc,
    )


def _run_encoding_mstcm_jax(hp, W, K, cat_indices, xp, dtype, e_start):
    """JAX scan-based encoding (parallels the numpy path exactly)."""
    import jax

    d = W + 1

    def step(carry, inputs):
        (c_global, c_list_g, c_item_per_s, c_story_per_s,
         m_fc_g, m_cf_g, m_fc_s, m_cf_s, m_sc, seen, prev_s) = carry
        t, s = inputs

        is_switch = (prev_s >= 0) & (s != prev_s)
        is_return = is_switch & seen[s]

        # Cache on switch.
        new_m_sc = xp.where(
            is_switch,
            at_set(m_sc, prev_s, c_story_per_s[prev_s, :], xp),
            m_sc,
        )

        # Decision 1A: λ reinstatement on return (BEFORE drift).
        cached = new_m_sc[s, :]
        current = c_story_per_s[s, :]
        blended = hp.lambda_reinstate * cached + (1.0 - hp.lambda_reinstate) * current
        n = xp.sqrt(xp.sum(blended * blended))
        blended_n = blended / xp.maximum(n, 1e-12)
        c_story_after_return = xp.where(is_return, blended_n, current)
        c_story_per_s = at_set(c_story_per_s, s, c_story_after_return, xp)

        # c^IN_t = e_{t+1}.
        c_in = xp.zeros(d, dtype=dtype)
        c_in = at_set(c_in, t + 1, 1.0, xp)

        # --- Pre-drift matrix updates ---
        m_fc_g = at_set(m_fc_g, (slice(None), t), m_fc_g[:, t] + c_global, xp)
        m_cf_g = at_set(m_cf_g, (t, slice(None)), m_cf_g[t, :] + c_global, xp)
        m_fc_s = at_set(
            m_fc_s, (s, slice(None), t),
            m_fc_s[s, :, t] + c_item_per_s[s, :], xp,
        )
        m_cf_s = at_set(
            m_cf_s, (s, t, slice(None)),
            m_cf_s[s, t, :] + c_item_per_s[s, :], xp,
        )

        # --- Drifts ---
        c_global = drift_c_ret(c_global, c_in, hp.beta_enc_global, xp)
        c_list_g = drift_c_ret(c_list_g, c_in, hp.beta_list, xp)
        c_item_per_s = at_set(
            c_item_per_s, s,
            drift_c_ret(c_item_per_s[s, :], c_in, hp.beta_enc, xp), xp,
        )
        c_story_per_s = at_set(
            c_story_per_s, s,
            drift_c_ret(c_story_per_s[s, :], c_in, hp.beta_list, xp), xp,
        )

        seen = at_set(seen, s, True, xp)
        return (
            (c_global, c_list_g, c_item_per_s, c_story_per_s,
             m_fc_g, m_cf_g, m_fc_s, m_cf_s, new_m_sc, seen, s),
            c_global,
        )

    # Initial state.
    c_global0 = e_start
    c_list_g0 = e_start
    c_item_per_s0 = xp.tile(e_start[None, :], (K, 1))
    c_story_per_s0 = xp.tile(e_start[None, :], (K, 1))
    m_fc_g0 = xp.zeros((d, W), dtype=dtype)
    m_cf_g0 = xp.zeros((W, d), dtype=dtype)
    m_fc_s0 = xp.zeros((K, d, W), dtype=dtype)
    m_cf_s0 = xp.zeros((K, W, d), dtype=dtype)
    m_sc0 = xp.zeros((K, d), dtype=dtype)
    seen0 = xp.zeros(K, dtype=bool)
    prev_s0 = xp.asarray(-1, dtype=xp.int32)

    init = (
        c_global0, c_list_g0, c_item_per_s0, c_story_per_s0,
        m_fc_g0, m_cf_g0, m_fc_s0, m_cf_s0, m_sc0, seen0, prev_s0,
    )
    t_arr = xp.arange(W, dtype=xp.int32)
    cat_indices_jax = xp.asarray(cat_indices, dtype=xp.int32)

    final, c_global_steps = jax.lax.scan(
        step, init, (t_arr, cat_indices_jax),
    )
    (c_global_f, c_list_g_f, c_item_per_s_f, c_story_per_s_f,
     m_fc_g_f, m_cf_g_f, m_fc_s_f, m_cf_s_f, m_sc_f, seen_f, prev_s_f) = final

    # Final cache update.
    m_sc_final = xp.where(
        prev_s_f >= 0,
        at_set(m_sc_f, prev_s_f, c_story_per_s_f[prev_s_f, :], xp),
        m_sc_f,
    )

    c_item_traj_g = xp.concatenate([e_start[None, :], c_global_steps], axis=0)
    return (
        c_item_traj_g, c_item_per_s_f, c_story_per_s_f, c_list_g_f,
        m_fc_g_f, m_cf_g_f, m_fc_s_f, m_cf_s_f, m_sc_final,
    )


# --- per-list log-likelihood --------------------------------------------


def compute_list_log_likelihood_mstcm(
    hp: MSCoreHyperparams,
    W: int,
    K: int,
    cat_indices,
    recall_sps,
    recall_mask,
    xp,
    dtype,
):
    """End-to-end per-list LL under MS-TCM (mixture over α/β routes).

    Computes:
        LL = log[ (1 − τ) · LL_α + τ · LL_β ]

    where:
    - LL_α: pure-global recall using M^CF_G, c_ret = c^global_end. Uses
      the standard C&Z stopping rule + drift dynamics. No storyline
      routing.
    - LL_β: strict-hierarchical recall. Determined by the observed
      sequence of storyline visits (cross-category transitions in the
      observed recalls reveal the storyline-visit boundaries). Each
      visit contributes:
        log P(ŝ_k | global cue, exclusion mask)
        + Σ within-storyline recall log-probs
        + log p_stop at end of visit (to fire the switch)
      The exclusion mask tracks ONLY the immediately-preceding storyline
      (per OQ2 resolution).

    Returns: scalar log-likelihood.
    """
    # Encode once; both routes share the same encoded representations.
    (c_item_traj_g, c_item_per_s, c_story_per_s, c_list_g_end,
     m_fc_g, m_cf_g, m_fc_s, m_cf_s, m_sc) = run_encoding_mstcm(
        hp, W, K, cat_indices, xp, dtype,
    )

    # Build the cat_per_position lookup: position -> storyline index.
    # In our model, item at serial position sp has 1-based sp, encoded
    # at step (sp-1), so its storyline = cat_indices[sp - 1].
    # (sp goes 1..W; cat_indices is 0-indexed; cat_indices[sp-1] is the
    # storyline of position sp.)

    # Compute LL for both routes (numpy path; the function is generic).
    if _is_jax(xp):
        # JAX path: write a single jit-compatible expression. For the
        # first iteration we delegate to the numpy reference under
        # `xp=numpy` then re-cast to JAX dtype. This is OK because
        # the numpy version uses xp-dispatched primitives so JAX inputs
        # would actually run through the same code paths if JAX were
        # passed in; the only issue is data-dependent control flow in
        # parsing the recall sequence into visits, which JAX cannot trace.
        # We mark this as a TODO; for now JAX backend uses the same numpy
        # control flow but xp-dispatched primitives.
        # For oracle tests we only need the numpy path; JAX support can
        # come once we add scan-based visit-parsing for jit compatibility.
        raise NotImplementedError(
            "JAX path for MS-TCM LL pending; numpy path is correct."
        )

    # numpy path: explicit visit-parsing.
    R_total = recall_sps.shape[0]

    # Route α: pure global recall (no storyline routing).
    ll_alpha = _ll_route_alpha_numpy(
        hp, W, c_ret_init=c_item_traj_g[W],
        m_fc_g=m_fc_g, m_cf_g=m_cf_g,
        recall_sps=recall_sps, recall_mask=recall_mask,
        xp=xp, dtype=dtype,
    )

    # Route β: strict hierarchical recall.
    ll_beta = _ll_route_beta_numpy(
        hp, W, K, cat_indices=cat_indices,
        c_global_end=c_item_traj_g[W],
        m_fc_g=m_fc_g, m_cf_g=m_cf_g,
        m_fc_s=m_fc_s, m_cf_s=m_cf_s,
        m_sc=m_sc,
        recall_sps=recall_sps, recall_mask=recall_mask,
        xp=xp, dtype=dtype,
    )

    # Mix the two routes.
    log_tau = xp.log(xp.maximum(hp.tau_init, 1e-300))
    log_1m_tau = xp.log(xp.maximum(1.0 - hp.tau_init, 1e-300))
    a = log_1m_tau + ll_alpha
    b = log_tau + ll_beta
    m = xp.maximum(a, b)
    return m + xp.log(xp.exp(a - m) + xp.exp(b - m))


# --- helper: route-α LL (pure global recall) ----------------------------


def _ll_route_alpha_numpy(
    hp, W, c_ret_init, m_fc_g, m_cf_g,
    recall_sps, recall_mask, xp, dtype,
):
    """Route α LL: standard global recall using M^CF_G.

    No hierarchical fallback (route α is "what C&Z would do without
    its e_start fallback"). LL = Σ [log(1-p_stop) + log p(sp)]
    + log p_stop_final.
    """
    d = c_ret_init.shape[0]
    R = recall_sps.shape[0]
    c_ret = xp.array(c_ret_init)
    cum_ll = 0.0
    already_mask = xp.zeros(W, dtype=bool)
    for i in range(R):
        sp = int(recall_sps[i])
        valid = bool(recall_mask[i])
        if not valid:
            continue
        idx = max(0, min(W - 1, sp - 1))
        is_repeat = bool(already_mask[idx])
        if is_repeat:
            continue  # treat repeats as noise (per pre-consolidation MS-TCM semantics)
        p_stop = float(compute_p_stop(
            m_cf_g, c_ret, already_mask, hp.epsilon_d, xp,
        ))
        log_1m = float(xp.log(max(1.0 - p_stop, 1e-300)))
        a = activation(m_cf_g, c_ret, xp)
        log_p = log_softmax_masked(hp.k * a, ~already_mask, xp)
        cum_ll += log_1m + float(log_p[idx])
        # Drift c_ret toward c^IN_rec(idx) at rate β_rec.
        c_in = c_IN_rec_of(idx, m_fc_g, hp.gamma_fc, d, xp, dtype)
        c_ret = drift_c_ret(c_ret, c_in, hp.beta_rec, xp)
        already_mask = at_set(already_mask, idx, True, xp)

    # Final stop.
    p_stop_final = float(compute_p_stop(
        m_cf_g, c_ret, already_mask, hp.epsilon_d, xp,
    ))
    cum_ll += float(xp.log(max(p_stop_final, 1e-300)))
    return cum_ll


# --- helper: route-β LL (strict hierarchical recall) --------------------


def _ll_route_beta_numpy(
    hp, W, K, cat_indices, c_global_end,
    m_fc_g, m_cf_g, m_fc_s, m_cf_s, m_sc,
    recall_sps, recall_mask, xp, dtype,
):
    """Route β LL: strict hierarchical recall.

    Parses the observed recalls into within-storyline visits (boundaries
    at observed category changes). For each visit:
        log P(ŝ_k | global, exclusion)  (exclusion = {ŝ_{k-1}}, single-int)
        + Σ within-storyline LL for that visit
        + log p_stop at end-of-visit (fires the storyline switch)
    For the LAST visit, the end-of-visit p_stop is the recall-terminator.
    """
    d = c_global_end.shape[0]
    R = recall_sps.shape[0]

    # --- Parse visits from observed recalls ---
    # Each visit is a contiguous block of recalls with the same storyline.
    # visits = list of (storyline_id, list_of_recall_indices_into_recall_sps).
    visits: list[tuple[int, list[int]]] = []
    current_storyline: int | None = None
    current_indices: list[int] = []
    for i in range(R):
        sp = int(recall_sps[i])
        valid = bool(recall_mask[i])
        if not valid:
            continue
        if sp == 0:
            continue  # intrusion
        idx_w = max(0, min(W - 1, sp - 1))
        s_of_recall = int(cat_indices[idx_w])
        if current_storyline is None or s_of_recall != current_storyline:
            # New visit boundary.
            if current_indices:
                visits.append((current_storyline, current_indices))
            current_storyline = s_of_recall
            current_indices = [i]
        else:
            current_indices.append(i)
    if current_indices:
        visits.append((current_storyline, current_indices))

    if not visits:
        # No valid recalls → no LL contribution from route β beyond a
        # single global stop. Use c_global_end with empty mask.
        empty_mask = xp.zeros(W, dtype=bool)
        # P(no storyline selected and stop immediately) is degenerate;
        # for pragmatic LL we treat as 0 contribution.
        return 0.0

    # --- Run through visits ---
    cum_ll = 0.0
    c_ret_g = xp.array(c_global_end)  # drifts every recall (Decision 2X)
    already_mask = xp.zeros(W, dtype=bool)
    last_visited_s = -1  # exclusion mask (single int, OQ2 resolution)

    for visit_idx, (s_hat, rec_indices) in enumerate(visits):
        # --- Storyline-selection log-prob ---
        # Build candidate mask: exclude `last_visited_s` only.
        candidate_mask = xp.ones(K, dtype=bool)
        if last_visited_s >= 0:
            candidate_mask = at_set(candidate_mask, last_visited_s, False, xp)
        if not candidate_mask[s_hat]:
            # The model says ŝ_hat is excluded but the observed recall
            # comes from there. This is a contradiction under the
            # exclusion mask. Return -inf LL (likelihood = 0).
            return float("-inf")
        # Softmax over candidates of (k * M^lists_G · c_ret_g).
        story_acts = m_sc @ c_ret_g  # (K,)
        log_p_story = log_softmax_masked(
            hp.k * story_acts, candidate_mask, xp,
        )
        cum_ll += float(log_p_story[s_hat])

        # --- Within-storyline recall LL ---
        # Cue: c_ret_s = e_start (option ii from OQ3).
        e_start = xp.zeros(d, dtype=dtype)
        e_start = at_set(e_start, 0, 1.0, xp)
        c_ret_s = e_start

        for j_in_visit, i in enumerate(rec_indices):
            sp = int(recall_sps[i])
            idx_w = max(0, min(W - 1, sp - 1))
            is_repeat = bool(already_mask[idx_w])
            if is_repeat:
                continue
            # Compute storyline-internal stopping rule using M^CF_s and
            # the storyline's own already-recalled mask.
            # The relevant items are those in storyline s_hat ONLY. The
            # already-mask within storyline s_hat = (already_mask[i] for
            # i where cat_indices[i]==s_hat).
            # However, our M^CF_s has shape (W, d), indexed by global
            # serial position. Items in OTHER storylines have zero rows
            # in M^CF_s (no associations were built for them in storyline
            # s_hat's matrix). So computing p_stop using M^CF_s and the
            # full W-length already_mask just naturally restricts to
            # storyline-s_hat items: the activations a_j for j in other
            # storylines are 0 (since M^CF_s[j] = 0), contributing nothing
            # to a_r or a_nr. But we do need to mask them as "not
            # available" for softmax. Solve by building an "in-storyline"
            # mask and combining.
            in_story_mask = xp.zeros(W, dtype=bool)
            for sp_idx in range(W):
                if int(cat_indices[sp_idx]) == s_hat:
                    in_story_mask = at_set(in_story_mask, sp_idx, True, xp)
            keep_mask = in_story_mask & ~already_mask

            p_stop = float(compute_p_stop(
                m_cf_s[s_hat], c_ret_s, already_mask | ~in_story_mask,
                hp.epsilon_d, xp,
            ))
            log_1m = float(xp.log(max(1.0 - p_stop, 1e-300)))
            a = activation(m_cf_s[s_hat], c_ret_s, xp)
            log_p = log_softmax_masked(hp.k * a, keep_mask, xp)
            cum_ll += log_1m + float(log_p[idx_w])

            # Drift Decision 2X: drift BOTH c_ret_s and c_ret_g.
            c_in_s = c_IN_rec_of(idx_w, m_fc_s[s_hat], hp.gamma_fc, d, xp, dtype)
            c_ret_s = drift_c_ret(c_ret_s, c_in_s, hp.beta_rec, xp)
            c_in_g = c_IN_rec_of(idx_w, m_fc_g, hp.gamma_fc, d, xp, dtype)
            c_ret_g = drift_c_ret(c_ret_g, c_in_g, hp.beta_rec, xp)
            already_mask = at_set(already_mask, idx_w, True, xp)

        # --- End-of-visit stop event ---
        # log p_stop at the storyline level (with the storyline's own
        # already-mask). This fires the visit boundary (or terminates
        # recall on the last visit).
        in_story_mask = xp.zeros(W, dtype=bool)
        for sp_idx in range(W):
            if int(cat_indices[sp_idx]) == s_hat:
                in_story_mask = at_set(in_story_mask, sp_idx, True, xp)
        p_stop_end = float(compute_p_stop(
            m_cf_s[s_hat], c_ret_s,
            already_mask | ~in_story_mask, hp.epsilon_d, xp,
        ))
        cum_ll += float(xp.log(max(p_stop_end, 1e-300)))

        # Update exclusion: only the just-finished storyline is excluded
        # for the NEXT global selection.
        last_visited_s = s_hat

    return cum_ll


# --- numpy entry point --------------------------------------------------


def compute_list_log_likelihood_mstcm_numpy(
    params,
    cat_indices,
    recall_sps,
    recall_mask,
    W: int,
    K: int,
) -> float:
    """Numpy entry point for MS-TCM list LL."""
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


# --- simulator ----------------------------------------------------------


def simulate_recalls_mstcm(
    params,
    W: int,
    K: int,
    cat_indices,
    rng,
    *,
    max_recalls: int | None = None,
) -> list[int]:
    """Sample a recall sequence from MS-TCM MS-TCM.

    Forward generative process:
      1. Encode → produce all contexts and matrices.
      2. Choose route α with prob (1 − τ) or route β with prob τ.
      3. Route α: standard pure-global recall using M^CF_G, c_ret =
         c^global_end. Drift c_ret toward c^IN_rec(i) at rate β_rec
         after each recall. Stop when stopping rule fires.
      4. Route β: repeat
            a. Sample storyline ŝ ∝ softmax(k · M^SC · c^global_end),
               excluding the immediately-preceding storyline.
            b. Set c_ret_s = e_start (option ii from OQ3).
            c. Sample within-storyline recalls from ŝ until stopping
               rule fires (uses M^CF_s and per-storyline already-mask).
            d. If no candidate storylines remain (all excluded), stop.
            e. Otherwise loop back to (a). After each full visit,
               update the immediately-preceding-storyline tracker.
    """
    import numpy as np

    hp = MSCoreHyperparams.from_model_parameters(params)
    d = W + 1
    cat = np.asarray(cat_indices, dtype=np.int64)

    # Encode.
    (c_item_traj_g, c_item_per_s, c_story_per_s, c_list_g_end,
     m_fc_g, m_cf_g, m_fc_s, m_cf_s, m_sc) = run_encoding_mstcm(
        hp, W, K, cat, np, np.float64,
    )

    cap = max_recalls if max_recalls is not None else 3 * W
    recalls: list[int] = []
    already_mask = np.zeros(W, dtype=bool)

    if rng.random() >= hp.tau_init:
        # --- Route α: pure global recall ---
        c_ret = c_item_traj_g[W].copy()
        while len(recalls) < cap:
            p_stop = float(compute_p_stop(
                m_cf_g, c_ret, already_mask, hp.epsilon_d, np,
            ))
            if rng.random() < p_stop:
                break
            a = activation(m_cf_g, c_ret, np)
            log_p = log_softmax_masked(hp.k * a, ~already_mask, np)
            p = np.exp(log_p - log_p.max())
            p = np.where(already_mask, 0.0, p)
            if p.sum() <= 0:
                break
            p = p / p.sum()
            idx = int(rng.choice(W, p=p))
            recalls.append(idx + 1)
            already_mask[idx] = True
            c_in = c_IN_rec_of(idx, m_fc_g, hp.gamma_fc, d, np, np.float64)
            c_ret = drift_c_ret(c_ret, c_in, hp.beta_rec, np)
        return recalls

    # --- Route β: strict hierarchical recall ---
    c_ret_g = c_item_traj_g[W].copy()
    last_visited_s = -1

    # Track FULLY-exhausted storylines (all items in that storyline have
    # been recalled). These are unconditionally excluded from selection,
    # not just for one iteration. This prevents an infinite loop where
    # the exclusion mask only blocks the most-recent storyline but the
    # model keeps selecting a storyline whose items are all already
    # recalled, then breaking out, then selecting again.
    fully_exhausted = np.zeros(K, dtype=bool)

    # Pre-compute in_story_mask per storyline.
    in_story_masks = np.zeros((K, W), dtype=bool)
    for sp_idx in range(W):
        in_story_masks[int(cat[sp_idx]), sp_idx] = True

    while len(recalls) < cap:
        # Update fully-exhausted: storyline s is exhausted if every
        # in_story item is in already_mask.
        for s in range(K):
            if not fully_exhausted[s]:
                if (in_story_masks[s] & ~already_mask).sum() == 0:
                    fully_exhausted[s] = True

        # Storyline selection: exclude fully-exhausted AND last-visited.
        candidate_mask = np.ones(K, dtype=bool)
        candidate_mask &= ~fully_exhausted
        if last_visited_s >= 0:
            candidate_mask[last_visited_s] = False
        # If only the last-visited storyline remains (because all others
        # are fully exhausted), allow it back as a fallback.
        if not candidate_mask.any():
            candidate_mask = ~fully_exhausted
            if not candidate_mask.any():
                break  # all storylines exhausted; recall ends.

        story_acts = m_sc @ c_ret_g  # (K,)
        log_p_story = log_softmax_masked(
            hp.k * story_acts, candidate_mask, np,
        )
        p_story = np.exp(log_p_story - log_p_story.max())
        p_story = np.where(candidate_mask, p_story, 0.0)
        if p_story.sum() <= 0:
            break
        p_story = p_story / p_story.sum()
        s_hat = int(rng.choice(K, p=p_story))

        # Within-storyline recall: c_ret_s = e_start (option ii).
        c_ret_s = np.zeros(d, dtype=np.float64)
        c_ret_s[0] = 1.0

        in_story_mask = in_story_masks[s_hat]

        # Track whether the visit produced any new recall (defensive
        # guard against pathological p_stop=1 cases that would otherwise
        # leave len(recalls) unchanged across visits).
        prev_recall_count = len(recalls)

        while len(recalls) < cap:
            keep_mask = in_story_mask & ~already_mask
            if not keep_mask.any():
                break  # all storyline-ŝ items already recalled
            p_stop = float(compute_p_stop(
                m_cf_s[s_hat], c_ret_s,
                already_mask | ~in_story_mask, hp.epsilon_d, np,
            ))
            if rng.random() < p_stop:
                break
            a = activation(m_cf_s[s_hat], c_ret_s, np)
            log_p = log_softmax_masked(hp.k * a, keep_mask, np)
            p = np.exp(log_p - log_p.max())
            p = np.where(keep_mask, p, 0.0)
            if p.sum() <= 0:
                break
            p = p / p.sum()
            idx = int(rng.choice(W, p=p))
            recalls.append(idx + 1)
            already_mask[idx] = True
            # Drift both contexts (Decision 2X).
            c_in_s = c_IN_rec_of(idx, m_fc_s[s_hat], hp.gamma_fc, d, np, np.float64)
            c_ret_s = drift_c_ret(c_ret_s, c_in_s, hp.beta_rec, np)
            c_in_g = c_IN_rec_of(idx, m_fc_g, hp.gamma_fc, d, np, np.float64)
            c_ret_g = drift_c_ret(c_ret_g, c_in_g, hp.beta_rec, np)

        # Defensive: if we just selected a storyline but produced no
        # new recall (e.g., immediate p_stop=1), mark it fully exhausted
        # to avoid re-selecting it in the next iteration.
        if len(recalls) == prev_recall_count:
            fully_exhausted[s_hat] = True

        last_visited_s = s_hat

    return recalls
