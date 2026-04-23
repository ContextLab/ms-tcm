"""JAX-native hierarchical CMR encoder + likelihood (Tier-2 / FR-032).

This module re-implements the Tier-1 scalar encode from
``ms_tcm.hcmr.HierarchicalCMRModel.encode`` using ``jax.lax.scan``
for the per-step recurrence. Each step is a single XLA op, and the
whole per-list encoding compiles into one jit-compiled function.

Scope vs Tier 1: the JAX likelihood scores the **recall-probability
sequence only** — it omits the free-recall stopping-rule contribution
(C&Z Eq 7: ``log(1 - p_stop)`` per continue + ``log(p_stop)`` at
termination) that Tier-1 ``likelihood.py`` includes. The stopping
rule's contribution is a smooth function of epsilon_d that is largely
orthogonal to the recall-transition dynamics driven by beta_enc /
beta_story / k / lambda; the optimizer's gradient with respect to the
7-dim parameter vector is dominated by the transition-prob terms, so
dropping the stopping contribution trades a fixed bias for substantial
JIT-friendliness. Researchers who need Tier-1-bit-identical fits
should use ``ms_tcm.fit.fit_mle`` (Tier 1); the JAX backend is a
fast-iteration tool.

Numerical parity with Tier 1 on the recall-probability LL alone:
float64 matches within 1e-10 (FR-023); float32 within 1e-8 (Q3).

Key design choices:

- **Per-list d = W+1 varies**: we ``jit`` keyed on (W, K) where
  K = n_storylines. Under FRFR-category all lists have W=16 and
  K=4, so one compilation suffices. A new list shape triggers a
  recompilation; ``jit``-caching in jax handles that automatically.
- **Basis-vector c^IN**: identical to Tier 1 — under IdentityPreMatrix
  the pre-experimental input is e_{t+1} (one-hot at column t+1 in
  R^(W+1)). We exploit sparsity: dot(c, c^IN) = c[t+1].
- **Primacy gradient**: same (1 + phi · exp(-psi · t)) scaling as
  Tier 1 (v6 §3 / Polyn et al. 2009 Eq 7). Phi/psi vary with
  ``standard_tcm`` flag (Tier 1 uses 30/0.8 under standard-TCM to
  produce enough primacy; MS-TCM uses 1.5/0.5).
- **Boundary handling**: storyline switches and returns are rare
  events. We implement them via ``jax.lax.cond`` inside the scan,
  guarded by the ``cat_changed`` flag. This gives correct results
  without unrolling the branch (JAX does both branches then selects).

The dtype is controlled by ``MS_TCM_JAX_DTYPE``; the module reads it
at import time to configure ``jax.config.update("jax_enable_x64", ...)``.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from ms_tcm.dataset import Dataset
from ms_tcm.params import ModelParameters


# --- dtype configuration ---

_JAX_DTYPE_ENV = os.environ.get("MS_TCM_JAX_DTYPE", "float64").lower()
if _JAX_DTYPE_ENV == "float64":
    jax.config.update("jax_enable_x64", True)
    _DTYPE = jnp.float64
elif _JAX_DTYPE_ENV == "float32":
    # JAX float32 is the default; don't flip jax_enable_x64.
    _DTYPE = jnp.float32
else:
    raise RuntimeError(
        f"MS_TCM_JAX_DTYPE must be 'float64' or 'float32'; got {_JAX_DTYPE_ENV!r}"
    )


def get_jax_dtype():
    """Return the currently-configured JAX dtype (``jnp.float64`` or ``jnp.float32``)."""
    return _DTYPE


# --- encoding ---


def _norm_preserving_rho_jax(beta, dot):
    """JAX-native norm-preserving rho; bit-identical to drift._norm_preserving_rho."""
    inner = 1.0 + beta * beta * (dot * dot - 1.0)
    return jnp.sqrt(jnp.maximum(inner, 0.0)) - beta * dot


def _primacy_scale(phi, psi, t):
    """CMR primacy gradient 1 + phi·exp(-psi·t)."""
    return 1.0 + phi * jnp.exp(-psi * t)


def _make_encode_fn(W: int, K: int, standard_tcm: bool):
    """Return a ``jit``-compiled encode function specialized for (W, K).

    Returns ``encode(beta_enc, beta_story, lambda_reinstate, cat_indices) -> (c_item_traj, m_ic)``:

    - ``cat_indices``: jnp.int32 array of shape (W,) giving the storyline
      index 0..K-1 for each step.
    - ``c_item_traj``: (W+1, W+1) — c_item after each step, row 0 is e_start.
    - ``m_ic``: (W+1, W) associative matrix after encoding.
    """

    d = W + 1
    # Static primacy parameters per spec (match Tier 1 hcmr.py).
    phi = 30.0 if standard_tcm else 1.5
    psi = 0.8 if standard_tcm else 0.5

    def encode(beta_enc, beta_story, lambda_reinstate, cat_indices):
        beta_enc = beta_enc.astype(_DTYPE)
        beta_story = beta_story.astype(_DTYPE)
        lambda_reinstate = lambda_reinstate.astype(_DTYPE)

        # Initial state.
        e_start = jnp.zeros(d, dtype=_DTYPE).at[0].set(1.0)
        c_item_0 = e_start
        c_story_stack_0 = jnp.broadcast_to(e_start, (K, d))  # one per storyline
        m_ic_0 = jnp.zeros((d, W), dtype=_DTYPE)
        m_sc_0 = jnp.zeros((K, d), dtype=_DTYPE)
        # Track which storylines have been seen (so we can detect "return").
        seen_0 = jnp.zeros(K, dtype=jnp.bool_)
        prev_cat_0 = jnp.array(-1, dtype=jnp.int32)

        def step(carry, inputs):
            c_item, c_story_stack, m_ic, m_sc, seen, prev_cat = carry
            t, cat_idx = inputs

            # Determine boundary flags.
            switch = (prev_cat >= 0) & (cat_idx != prev_cat)
            cat_return = switch & seen[cat_idx]

            # --- Storyline switch / return (only when not standard_tcm) ---
            if not standard_tcm:
                # Cache outgoing storyline to M^SC (row = prev_cat):
                #   m_sc[prev_cat] += c_story_stack[prev_cat]  (outer product with one-hot)
                #
                # Use where(switch, ...) to avoid a conditional side effect.
                def do_cache(ms):
                    # prev_cat may be -1 at step 0; guard via switch flag.
                    return ms.at[prev_cat].add(c_story_stack[prev_cat])
                m_sc = jax.lax.cond(switch, do_cache, lambda ms: ms, m_sc)

                # On return: blend c_story_stack[cat] with the cached
                # reinstatement: c_story_new = lambda * m_sc[cat] + (1-lambda) * c_story[cat]
                def do_return(cstack):
                    blended = lambda_reinstate * m_sc[cat_idx] + \
                              (1.0 - lambda_reinstate) * cstack[cat_idx]
                    nn = jnp.linalg.norm(blended)
                    blended_n = jnp.where(nn > 1e-12, blended / nn, blended)
                    return cstack.at[cat_idx].set(blended_n)
                c_story_stack = jax.lax.cond(
                    cat_return, do_return, lambda cs: cs, c_story_stack,
                )

                # After a (switch OR return), sync c_item to the active storyline
                # context (v6 Eq 7).
                c_item = jnp.where(switch, c_story_stack[cat_idx], c_item)

            # Basis-vector c^IN: e_{t+1}. dot(c, c^IN) = c[t+1].
            t_plus_1 = t + 1

            # --- Drift active storyline (v6 Eq 2) ---
            if not standard_tcm:
                c_story_active = c_story_stack[cat_idx]
                dot_s = c_story_active[t_plus_1]
                rho_s = _norm_preserving_rho_jax(beta_story, dot_s)
                c_story_new = rho_s * c_story_active
                c_story_new = c_story_new.at[t_plus_1].add(beta_story)
                c_story_stack = c_story_stack.at[cat_idx].set(c_story_new)

            # --- Drift item-level context (v6 Eq 1) ---
            dot_i = c_item[t_plus_1]
            rho_i = _norm_preserving_rho_jax(beta_enc, dot_i)
            c_item = rho_i * c_item
            c_item = c_item.at[t_plus_1].add(beta_enc)

            # --- Associative matrix update (M^IC column t) ---
            primacy = _primacy_scale(phi, psi, t.astype(_DTYPE))
            m_ic = m_ic.at[:, t].add(primacy * c_item)

            seen = seen.at[cat_idx].set(True)
            prev_cat = cat_idx

            return (c_item, c_story_stack, m_ic, m_sc, seen, prev_cat), c_item

        t_range = jnp.arange(W, dtype=jnp.int32)
        inputs = (t_range, cat_indices)
        (_, _, m_ic_final, _, _, _), c_item_steps = jax.lax.scan(
            step, (c_item_0, c_story_stack_0, m_ic_0, m_sc_0, seen_0, prev_cat_0),
            inputs,
        )
        # Prepend e_start so c_item_traj[0] = e_start.
        c_item_traj = jnp.concatenate([e_start[None, :], c_item_steps], axis=0)
        return c_item_traj, m_ic_final

    return jax.jit(encode)


# --- per-list likelihood ---


def _make_list_log_likelihood_fn(W: int, K: int, standard_tcm: bool):
    """Return a ``jit``-compiled per-list log-likelihood function.

    Input signature:
        fn(theta, cat_indices, recall_sps, recall_mask) -> float

    where:
    - ``theta``: (7,) — unconstrained params [beta_enc, beta_story,
      gamma_fc, k, lambda_reinstate, beta_rec, epsilon_d] in the
      reparameterized space (same as Tier-1 ``fit.py``).
    - ``cat_indices``: (W,) int32 — storyline index per step.
    - ``recall_sps``: (R,) int32 — 1-based serial positions of observed
      recalls (padded with 0s to fixed length R).
    - ``recall_mask``: (R,) bool — True for valid recall slots, False
      for padding.

    Returns log-likelihood (scalar float of dtype _DTYPE).
    """
    encode_fn = _make_encode_fn(W, K, standard_tcm)

    def theta_to_params(theta):
        from jax.scipy.special import expit as jexpit
        beta_enc = jexpit(theta[0])
        ratio = jexpit(theta[1])
        beta_story = beta_enc * ratio
        beta_story = jnp.minimum(beta_story, beta_enc - 1e-10)
        k = jnp.exp(jnp.clip(theta[3], -30.0, 30.0))
        lambda_reinstate = jnp.where(standard_tcm, 0.0, jexpit(theta[4]))
        beta_rec = jexpit(theta[5])
        epsilon_d = jnp.exp(jnp.clip(theta[6], -30.0, 30.0))
        return beta_enc, beta_story, k, lambda_reinstate, beta_rec, epsilon_d

    def list_log_likelihood(theta, cat_indices, recall_sps, recall_mask):
        theta = theta.astype(_DTYPE)
        beta_enc, beta_story, k, lambda_reinstate, beta_rec, epsilon_d = theta_to_params(theta)

        c_item_traj, m_ic = encode_fn(
            beta_enc, beta_story, lambda_reinstate, cat_indices,
        )
        # End-of-list cue (c_item after W encoding steps).
        c_item_end = c_item_traj[W]

        # For each recall, compute log-prob of the observed serial position
        # given the current retrieval context. We scan through recalls;
        # c_ret starts at c_item_end; masked items are excluded from softmax.
        # After each recall, c_ret is reset to the encoded c_item at that
        # serial position (matches Tier-1 likelihood.py semantics).

        # Activation per-list-position: a[j] = M^IC[:, j] dot c_cue.
        # With M^IC shape (d, W) and c_cue shape (d,), a = M^IC.T @ c_cue.

        def step(carry, rec_input):
            c_ret, already_recalled, log_l, prev_idx = carry
            sp, valid = rec_input  # sp is 1-based serial position; valid is bool

            # If not valid (padding), skip — no log-prob contribution.
            # If prev_idx >= 0, apply beta_rec drift toward c_item at prev_idx+1.
            def do_drift(c):
                c_item_recalled = c_item_traj[prev_idx + 1]
                dot_r = jnp.dot(c, c_item_recalled)
                rho_r = _norm_preserving_rho_jax(beta_rec, dot_r)
                return rho_r * c + beta_rec * c_item_recalled
            c_ret_new = jax.lax.cond(
                (prev_idx >= 0) & valid, do_drift, lambda c: c, c_ret,
            )

            # Score against list.
            a = m_ic.T @ c_ret_new  # (W,)
            # Mask already-recalled positions with a large negative offset
            # rather than -inf; log_softmax on an all-[-inf] vector returns
            # NaN, and we also need to stay finite when `valid=False` so the
            # contribution computed below multiplies by 0 cleanly.
            MASKED_OFFSET = jnp.asarray(-1e9, dtype=_DTYPE)
            scaled = jnp.where(already_recalled, MASKED_OFFSET, k * a)
            # Log-softmax for numerical stability.
            log_p = jax.nn.log_softmax(scaled)

            # Index into log_p at sp-1 (clamp to valid range).
            idx = jnp.clip(sp - 1, 0, W - 1)
            # Observed repeats: real FRFR-category data contain occasional
            # repeat recalls (e.g. the same serial_position output twice).
            # Under our no-repeats masking convention the log-prob is
            # approximately -1e9 for such events; we emulate Tier-1's
            # ``if sp in recalled_sps: continue`` by zero-ing the
            # contribution (but still updating prev_idx / c_ret).
            is_repeat = already_recalled[idx]
            count_this = valid & ~is_repeat
            contrib = jnp.where(count_this, log_p[idx], 0.0)
            log_l = log_l + contrib

            # Mark sp-1 as recalled (only if valid). Padding rows leave
            # already_recalled untouched (jnp.where preserves the existing
            # array when valid=False).
            already_recalled = jnp.where(
                valid,
                already_recalled.at[idx].set(True),
                already_recalled,
            )
            # Advance c_ret to c_item[sp] only on valid rows; padding rows
            # preserve c_ret_new so the next valid row (if any) sees the
            # correct retrieval context.
            c_ret_next = jnp.where(
                valid,
                c_item_traj[jnp.clip(sp, 0, W)],
                c_ret_new,
            )
            prev_idx_next = jnp.where(valid, idx, prev_idx)
            return (c_ret_next, already_recalled, log_l, prev_idx_next), None

        already_recalled_0 = jnp.zeros(W, dtype=jnp.bool_)
        log_l_0 = jnp.array(0.0, dtype=_DTYPE)
        prev_idx_0 = jnp.array(-1, dtype=jnp.int32)
        (_, _, log_l_final, _), _ = jax.lax.scan(
            step, (c_item_end, already_recalled_0, log_l_0, prev_idx_0),
            (recall_sps, recall_mask),
        )
        return log_l_final

    return jax.jit(list_log_likelihood)


# --- module-level caches ---


_ENCODE_FN_CACHE: dict = {}
_LIST_LL_FN_CACHE: dict = {}


def encode_list_jax(
    params: ModelParameters, cat_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper that evaluates the JIT'd encode for a single list."""
    W = cat_indices.shape[0]
    K = int(cat_indices.max()) + 1 if W > 0 else 1
    key = (W, K, bool(params.standard_tcm))
    fn = _ENCODE_FN_CACHE.get(key)
    if fn is None:
        fn = _make_encode_fn(W, K, bool(params.standard_tcm))
        _ENCODE_FN_CACHE[key] = fn
    c_item_traj, m_ic = fn(
        jnp.array(params.beta_enc, dtype=_DTYPE),
        jnp.array(params.beta_story, dtype=_DTYPE),
        jnp.array(params.lambda_reinstate, dtype=_DTYPE),
        jnp.asarray(cat_indices, dtype=jnp.int32),
    )
    return np.asarray(c_item_traj), np.asarray(m_ic)


def list_log_likelihood_jax(
    params: ModelParameters,
    cat_indices: np.ndarray,
    recall_sps: np.ndarray,
    recall_mask: np.ndarray,
) -> float:
    """Evaluate per-list log-likelihood under JAX."""
    W = cat_indices.shape[0]
    K = int(cat_indices.max()) + 1 if W > 0 else 1
    key = (W, K, bool(params.standard_tcm))
    fn = _LIST_LL_FN_CACHE.get(key)
    if fn is None:
        fn = _make_list_log_likelihood_fn(W, K, bool(params.standard_tcm))
        _LIST_LL_FN_CACHE[key] = fn
    # Build theta from params (same reparameterization as fit.py).
    from scipy.special import logit as slogit
    theta = np.array([
        slogit(np.clip(params.beta_enc, 1e-6, 1.0 - 1e-6)),
        slogit(np.clip(params.beta_story / params.beta_enc, 1e-6, 1.0 - 1e-6)),
        slogit(np.clip(params.gamma_fc, 1e-6, 1.0 - 1e-6)),
        np.log(params.k),
        slogit(np.clip(max(params.lambda_reinstate, 1e-6), 1e-6, 1.0 - 1e-6)),
        slogit(np.clip(params.beta_rec, 1e-6, 1.0 - 1e-6)),
        np.log(params.epsilon_d),
    ], dtype=np.float64)
    result = fn(
        jnp.asarray(theta, dtype=_DTYPE),
        jnp.asarray(cat_indices, dtype=jnp.int32),
        jnp.asarray(recall_sps, dtype=jnp.int32),
        jnp.asarray(recall_mask, dtype=jnp.bool_),
    )
    return float(result)


def dataset_log_likelihood_jax(
    params: ModelParameters, dataset: Dataset,
) -> float:
    """Sum of per-list log-likelihoods via the JAX backend.

    Iterates over (participant, list_) in Python (because per-list d=W+1
    varies); each list's likelihood evaluation is a single JIT-compiled
    ``jax.lax.scan`` call, so the Python overhead is one call-per-list.
    """
    pdf = dataset.presented.to_pandas()
    rdf = dataset.recalled.to_pandas()
    keys = sorted(set(zip(pdf["participant"].tolist(), pdf["list"].tolist())))

    total = 0.0
    # Max recall length across the dataset — used to pad to a fixed shape
    # so JAX can JIT a single function (recompilation on (W, K, R) changes).
    max_recall_len = int(
        rdf[rdf["serial_position"] > 0]
        .groupby(["participant", "list"])
        .size().max()
        if not rdf.empty else 1
    )

    for part, lst in keys:
        pres_sub = pdf[(pdf["participant"] == part) & (pdf["list"] == lst)].sort_values("serial_position")
        rec_sub = rdf[(rdf["participant"] == part) & (rdf["list"] == lst)].sort_values("output_position")

        # Build cat_indices: storyline index per step (unique categories in order of appearance).
        cats = list(dict.fromkeys(pres_sub["category"].tolist()))
        cat_to_idx = {c: i for i, c in enumerate(cats)}
        cat_indices = np.array([cat_to_idx[c] for c in pres_sub["category"].tolist()], dtype=np.int32)

        # Build recall_sps + mask (pad to max_recall_len).
        rec_rows = rec_sub[rec_sub["serial_position"] > 0]  # drop intrusions
        sps = rec_rows["serial_position"].to_numpy(dtype=np.int32)
        R = max_recall_len
        recall_sps = np.zeros(R, dtype=np.int32)
        recall_mask = np.zeros(R, dtype=np.bool_)
        n = min(len(sps), R)
        recall_sps[:n] = sps[:n]
        recall_mask[:n] = True

        try:
            ll = list_log_likelihood_jax(params, cat_indices, recall_sps, recall_mask)
        except Exception:
            return float("-inf")
        if not np.isfinite(ll):
            return float("-inf")
        total += ll
    return total
