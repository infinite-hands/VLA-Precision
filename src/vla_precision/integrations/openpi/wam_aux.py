"""WAM-style auxiliary future-prediction loss (see docs/wam-aux-loss.md): the real, JAX/Flax
NNX training-loop port of the Phase 0 offline probe's predictor
(vla_precision/aux_probe_predictor.py in the main infinite-hands repo, branch
feat/wam-embedding-probe). Phase 0 needed its own `_ContextEncoder` (a small bidirectional
self-attention stack) plus an EMA copy of it, because it operated on cached embeddings from a
frozen checkpoint it had no other way to update or track through time. Here, "now" (z_t) and
"future" (z_future) are Pi0's OWN right-wrist prefix embeddings -- already produced by a real
bidirectional self-attention stack (SigLIP/PaliGemma's vision tower), and already available in
two forms: "online" (the live, being-trained params) and "EMA-stabilized" (the model reconstructed
from `state.ema_params`, openpi's own existing checkpoint-quality EMA machinery, reused here as
the BYOL/JEPA-style target encoder). So this module ports only the parts Pi0 doesn't already
supply: the predictor itself (a learned positional query per output position, refined by
cross-attention blocks conditioned on flow-matching time / offset k / the action window t..t+k),
its own GRU action-sequence encoder, and one small EMA-lagged down-projection (see
`AuxProjection` below) -- the one new, freshly-initialized layer in this whole design, and
therefore the one place a fresh EMA lag is actually needed to preserve the asymmetry a
self-predictive loss requires to avoid collapsing to a constant.

Flow-matching noise/time convention is ported byte-for-byte from the validated Phase 0 code
(uniform tau in [0, 1], `noised = (1 - tau) * noise + tau * target`, velocity target
`target - noise`) rather than pi0.5's own Beta(1.5, 1) convention -- the two aren't the same
mechanism despite the plan doc's original aspiration to match pi0.5's; this ports what was
actually trained and validated on real data, not the earlier aspiration.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct

TOKEN_DIM = 2048  # PaliGemma hidden size; Pi0.embed_prefix's per-token output width
NUM_TOKENS = 256  # confirmed token count for one 224x224 SigLIP camera view (Phase 0 extraction)
D_MODEL = 384
PREDICTOR_DEPTH = 2
HEADS = 6
MLP_RATIO = 4
AUX_EMA_MOMENTUM = 0.99  # BYOL's standard starting point; the user's explicit choice for this loss's own new encoder (distinct from the backbone's own ema_decay, which serves checkpoint-quality EMA and stays at whatever Stage1Config.ema_decay already is)


def _timestep_embedding(t: jax.Array, dim: int) -> jax.Array:
    """Sinusoidal embedding, ported byte-for-byte from aux_probe_predictor.py's
    `timestep_embedding` (itself matching custom_model/model.py's flow-time embedding
    convention) -- reused for both the flow time tau and the prediction offset k."""
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(dim // 2, dtype=jnp.float32) / (dim // 2))
    angles = t.astype(jnp.float32)[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1)


class AuxProjection(nnx.Module):
    """The one new, freshly-initialized layer in this design (token_dim -> d_model), and
    therefore the one place that needs its own EMA lag: an aux loss comparing the ONLINE
    projection's output against the SAME (shared-weight, non-lagged) projection's output on the
    EMA-encoder's tokens would reintroduce exactly the collapse risk BYOL's target lag exists to
    prevent, since nothing about `state.ema_params`'s backbone-level lag protects a layer that
    only exists downstream of it. `target` is a plain, non-nnx copy of this module's own params,
    updated via `update_ema` after each optimizer step (BYOL's ordering) -- never touched by the
    optimizer itself, and never in `trainable_filter`/the diffed argnum."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.online = nnx.Linear(TOKEN_DIM, D_MODEL, rngs=rngs)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        return self.online(tokens)


@struct.dataclass
class _AuxProjectionEma:
    """A plain (non-nnx) pytree holding an EMA-lagged copy of AuxProjection's params, mirroring
    how `state.ema_params` already shadows `state.params` for the backbone -- kept as its own
    small pytree, not nnx state, since it never needs to be an argument `nnx.value_and_grad`
    differentiates. `flax.struct.dataclass` (not a plain dataclass) so it can be nested inside
    `AuxState` and flow through `jax.jit` -- the same reason `training_utils.TrainState` itself
    uses `flax.struct` rather than a plain `@dataclasses.dataclass`."""

    kernel: jax.Array
    bias: jax.Array

    @classmethod
    def from_projection(cls, projection: AuxProjection) -> "_AuxProjectionEma":
        return cls(kernel=jnp.array(projection.online.kernel.value), bias=jnp.array(projection.online.bias.value))

    def apply(self, tokens: jax.Array) -> jax.Array:
        return tokens @ self.kernel + self.bias

    def update(self, projection: AuxProjection, momentum: float = AUX_EMA_MOMENTUM) -> "_AuxProjectionEma":
        return _AuxProjectionEma(
            kernel=momentum * self.kernel + (1 - momentum) * projection.online.kernel.value,
            bias=momentum * self.bias + (1 - momentum) * projection.online.bias.value,
        )


def _standardize(x: jax.Array) -> jax.Array:
    """Zero-mean, unit-variance per token (LayerNorm without learnable parameters).

    Without this the auxiliary loss is a raw MSE in a representation space nothing constrains the
    scale of, and the loss inflates mechanically as the backbone trains rather than because
    prediction actually got worse: the EMA target's magnitude tracks the online backbone's, so
    once embedding magnitudes start drifting upward, so does the loss -- and, because gradient
    reaches the backbone through `context`, so does the gradient this loss pushes into it.
    Measured on a real 500-step run (see docs/wam-aux-loss.md): aux_loss fell 16.9 -> 2.2 over the
    first 40 steps, then climbed monotonically back to 15.1 while the primary gradient norm went
    from ~0.3 to spikes of 4.9-6.9. Reproduced in isolation with a synthetic inflating input scale
    and eliminated by this normalization (climb ratio 2.31x -> 1.00x).

    Phase 0's offline probe did not need this because its backbone was FROZEN, so target scale was
    constant by construction; that assumption silently broke once the loss trained jointly with a
    live backbone. Normalizing both sides is what BYOL/JEPA implementations do for this reason, and
    it costs the per-token magnitude as a predictable signal -- an accepted trade for a bounded,
    well-scaled objective."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + 1e-6)


def _cosine_similarity(a: jax.Array, b: jax.Array) -> jax.Array:
    """Per-token cosine similarity along the feature axis, returning (..., num_tokens)."""
    a_norm = a * jax.lax.rsqrt(jnp.sum(jnp.square(a), axis=-1, keepdims=True) + 1e-8)
    b_norm = b * jax.lax.rsqrt(jnp.sum(jnp.square(b), axis=-1, keepdims=True) + 1e-8)
    return jnp.sum(a_norm * b_norm, axis=-1)


def _adaln_zero_linear(cond_dim: int, out_dim: int, *, rngs: nnx.Rngs) -> nnx.Linear:
    """A Linear whose output starts at exactly zero (adaLN-zero) so every predictor block begins
    training as an identity residual and only gradually turns on -- ported from
    aux_probe_predictor.py's `_CrossAttnPredictorBlock.__init__`, which zero-inits the same way."""
    return nnx.Linear(cond_dim, out_dim, kernel_init=nnx.initializers.zeros, bias_init=nnx.initializers.zeros, rngs=rngs)


class _CrossAttnPredictorBlock(nnx.Module):
    """Per output position: bidirectional self-attention among the query tokens (predicted future
    positions are correlated with each other), THEN cross-attention to the context tokens as K/V
    (the positional query "asks" the context what it should become). Both adaLN-zero conditioned
    on (k, tau, action-window). Ported from aux_probe_predictor.py's `_CrossAttnPredictorBlock`."""

    def __init__(self, d_model: int, cond_dim: int, heads: int, mlp_ratio: int, *, rngs: nnx.Rngs):
        self.norm_self = nnx.LayerNorm(d_model, use_bias=False, use_scale=False, epsilon=1e-6, rngs=rngs)
        self.self_attn = nnx.MultiHeadAttention(heads, d_model, decode=False, rngs=rngs)
        self.norm_cross = nnx.LayerNorm(d_model, use_bias=False, use_scale=False, epsilon=1e-6, rngs=rngs)
        self.cross_attn = nnx.MultiHeadAttention(heads, d_model, decode=False, rngs=rngs)
        self.norm_mlp = nnx.LayerNorm(d_model, use_bias=False, use_scale=False, epsilon=1e-6, rngs=rngs)
        self.mlp_in = nnx.Linear(d_model, d_model * mlp_ratio, rngs=rngs)
        self.mlp_out = nnx.Linear(d_model * mlp_ratio, d_model, rngs=rngs)
        self.modulation = _adaln_zero_linear(cond_dim, 9 * d_model, rngs=rngs)

    def __call__(self, query: jax.Array, context: jax.Array, condition: jax.Array) -> jax.Array:
        mod = self.modulation(nnx.silu(condition))[:, None, :]
        s1, c1, g1, s2, c2, g2, s3, c3, g3 = jnp.split(mod, 9, axis=-1)

        normed = self.norm_self(query) * (1 + c1) + s1
        query = query + g1 * self.self_attn(normed)

        normed = self.norm_cross(query) * (1 + c2) + s2
        query = query + g2 * self.cross_attn(normed, context, context)

        normed = self.norm_mlp(query) * (1 + c3) + s3
        query = query + g3 * self.mlp_out(nnx.gelu(self.mlp_in(normed)))
        return query


class _ActionSequenceEncoder(nnx.Module):
    """A GRU over the action window t:t+k -- order-aware, unlike a mean-pool over the window.
    Ported from aux_probe_predictor.py's `_ActionSequenceEncoder`."""

    def __init__(self, action_dim: int, d_model: int, *, rngs: nnx.Rngs):
        self.proj = nnx.Linear(action_dim, d_model, rngs=rngs)
        self.cell = nnx.GRUCell(d_model, d_model, rngs=rngs)
        self.d_model = d_model

    def __call__(self, actions: jax.Array) -> jax.Array:
        # actions: (batch, k, action_dim). lax.scan needs the scanned axis leading, so the
        # projected window is transposed to (k, batch, d_model); the GRU's per-step call is
        # unaffected by this since nnx.Linear/GRUCell both operate on the trailing feature axis.
        projected = self.proj(actions)
        time_major = jnp.swapaxes(projected, 0, 1)
        carry = jnp.zeros((actions.shape[0], self.d_model), dtype=projected.dtype)

        def step(carry, x):
            return self.cell(carry, x)

        final_carry, _ = jax.lax.scan(step, carry, time_major)
        return final_carry  # (batch, d_model): the GRU's final hidden state, order-aware


class AuxFuturePredictor(nnx.Module):
    """context (z_t, online projection) + a content-free learned positional query per output
    position, refined by `_CrossAttnPredictorBlock`s conditioned on (k, action-window) into a
    DIRECT prediction of the future embedding, in `AuxProjection`'s d_model space.

    The queries carry no content: each output position asks the context what it should become,
    which is exactly I-JEPA/V-JEPA's mask-token formulation. An earlier version instead seeded each
    query with a noised sample of the target and predicted a flow-matching velocity; see
    `compute_aux_loss` for why that was removed."""

    def __init__(
        self,
        *,
        action_dim: int,
        num_tokens: int = NUM_TOKENS,
        d_model: int = D_MODEL,
        heads: int = HEADS,
        depth: int = PREDICTOR_DEPTH,
        mlp_ratio: int = MLP_RATIO,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.query_pos_embed = nnx.Param(nnx.initializers.normal(0.02)(rngs.params(), (1, num_tokens, d_model)))
        # Position must be identifiable on the CONTEXT side too, not just the query side. The
        # queries are content-free (V-JEPA's mask-token formulation), so the query for output
        # position i has to locate context token i among all 256 by cross-attention -- and it
        # cannot match on content, because the content is exactly what it is trying to predict.
        # Without this, cross-attention has no positional handle to match on and the predictor
        # cannot even solve the oracle task (measured: loss stuck at 0.98-0.99 for 1500 steps with
        # context set to the literal answer). An earlier flow-matching version accidentally avoided
        # this by seeding each query with a noised sample of its own target, which anchored it.
        self.context_pos_embed = nnx.Param(nnx.initializers.normal(0.02)(rngs.params(), (1, num_tokens, d_model)))
        self.action_encoder = _ActionSequenceEncoder(action_dim, d_model, rngs=rngs)
        self.condition_mlp_in = nnx.Linear(d_model, d_model, rngs=rngs)
        self.condition_mlp_out = nnx.Linear(d_model, d_model, rngs=rngs)
        self.blocks = [_CrossAttnPredictorBlock(d_model, d_model, heads, mlp_ratio, rngs=rngs) for _ in range(depth)]
        self.out_norm = nnx.LayerNorm(d_model, use_bias=False, use_scale=False, epsilon=1e-6, rngs=rngs)
        # NOT zero-initialized, unlike the adaLN gates: the loss is a cosine, and an exactly-zero
        # prediction has no direction, so its cosine against the target is 0/0. The adaLN-zero
        # gates still make every block an identity at init, so the warm start is unchanged -- the
        # prediction simply begins as a fixed function of the positional queries, which is
        # uncorrelated with the target and therefore reads as the loss's natural 1.0 baseline.
        self.out = nnx.Linear(d_model, d_model, rngs=rngs)

    def __call__(self, context: jax.Array, *, k: jax.Array, actions: jax.Array) -> jax.Array:
        # Seeded from the context at the SAME position, not a content-free constant. Cross-attention
        # alone cannot carry position: at init the softmax is near-uniform, so every output position
        # receives the same pooled summary of the context and scores cosine ~0 against its own
        # target, and escaping that requires learning a near-one-hot "attend to myself" pattern
        # first. Measured on the oracle task (context = the literal answer), content-free queries
        # stayed at 0.99 for 800 steps while a bare per-position linear readout reached 0.0005 --
        # the routing, not the objective, was the bottleneck. Seeding this way also makes the
        # predictor's job the natural one: predict the CHANGE from present to future. That does
        # make the identity map an easy solution, which is exactly what `copy_baseline` in
        # compute_aux_loss measures, so a predictor that only learns to copy is visible rather than
        # mistaken for success.
        query = context + self.query_pos_embed
        context = context + self.context_pos_embed
        action_emb = self.action_encoder(actions)
        k_emb = _timestep_embedding(k.astype(jnp.float32), self.d_model)
        condition = self.condition_mlp_out(nnx.silu(self.condition_mlp_in(action_emb + k_emb)))
        for block in self.blocks:
            query = block(query, context, condition)
        return self.out(self.out_norm(query))


@struct.dataclass
class AuxState:
    """The auxiliary loss's own, separate optimizer state -- kept as a sibling to openpi's own
    `training_utils.TrainState`, not merged into it, so the backbone's own state/checkpoint/
    sharding plumbing (all upstream openpi code) stays completely untouched. Only present when
    `aux_loss_weight > 0`; `_train()` skips constructing this entirely otherwise.

    `*_params` holds the FULL nnx state (every variable, not just `nnx.Param` leaves) -- mirrors
    how `training_utils.TrainState.params` itself is the full `nnx.state(model)`, with
    `.filter(trainable_filter)` applied only at the point of optimizer use (`train.py`'s
    `train_step`). Storing the unfiltered state here and filtering the same way at update time
    keeps this file's optax plumbing structurally consistent with train.py's own, rather than a
    parallel convention that happens to look similar.

    `flax.struct.dataclass`, matching `training_utils.TrainState` exactly (down to marking `tx`
    `pytree_node=False`, the same field for the same reason: an `optax.GradientTransformation` is
    a tuple of functions, not array data) -- required for this to flow through `jax.jit` at all."""

    projection_graphdef: nnx.GraphDef
    projection_params: nnx.State
    projection_ema: _AuxProjectionEma
    predictor_graphdef: nnx.GraphDef
    predictor_params: nnx.State
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)


def init_aux_state(*, action_dim: int, learning_rate: float, rngs: nnx.Rngs) -> AuxState:
    projection = AuxProjection(rngs=rngs)
    predictor = AuxFuturePredictor(action_dim=action_dim, rngs=rngs)
    projection_graphdef, projection_params = nnx.split(projection)
    predictor_graphdef, predictor_params = nnx.split(predictor)
    tx = optax.adam(learning_rate)
    opt_state = tx.init(
        (projection_params.filter(nnx.All(nnx.Param)), predictor_params.filter(nnx.All(nnx.Param)))
    )
    return AuxState(
        projection_graphdef=projection_graphdef,
        projection_params=projection_params,
        projection_ema=_AuxProjectionEma.from_projection(projection),
        predictor_graphdef=predictor_graphdef,
        predictor_params=predictor_params,
        opt_state=opt_state,
        tx=tx,
    )


def apply_aux_update(
    aux_state: AuxState, grads_projection: nnx.State, grads_predictor: nnx.State
) -> AuxState:
    """One optimizer step for the aux predictor + its projection, then the BYOL EMA update on the
    projection (after the step, matching aux_probe_predictor.py's `update_ema()` docstring: "call
    AFTER each optimizer step, so the target reflects the just-updated online weights"). Mirrors
    train.py's own train_step update block line-for-line, just applied to this sibling state
    instead of `training_utils.TrainState`."""
    projection = nnx.merge(aux_state.projection_graphdef, aux_state.projection_params)
    predictor = nnx.merge(aux_state.predictor_graphdef, aux_state.predictor_params)

    proj_trainable = aux_state.projection_params.filter(nnx.All(nnx.Param))
    pred_trainable = aux_state.predictor_params.filter(nnx.All(nnx.Param))
    updates, new_opt_state = aux_state.tx.update(
        (grads_projection, grads_predictor), aux_state.opt_state, (proj_trainable, pred_trainable)
    )
    new_proj_trainable = optax.apply_updates(proj_trainable, updates[0])
    new_pred_trainable = optax.apply_updates(pred_trainable, updates[1])

    nnx.update(projection, new_proj_trainable)
    nnx.update(predictor, new_pred_trainable)

    return dataclasses.replace(
        aux_state,
        projection_params=nnx.state(projection),
        projection_ema=aux_state.projection_ema.update(projection),
        predictor_params=nnx.state(predictor),
        opt_state=new_opt_state,
    )


def compute_aux_loss(
    projection: AuxProjection,
    predictor: AuxFuturePredictor,
    projection_ema: _AuxProjectionEma,
    rng: jax.Array,
    *,
    context_tokens: jax.Array,
    target_tokens_raw: jax.Array,
    offset_k: int,
    action_window: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """WAM auxiliary future-prediction loss (see docs/wam-aux-loss.md): mean per-token cosine
    distance between the predicted and the EMA-encoded future embedding, which is what JEPA-WAM
    (arXiv 2608.09381) uses for exactly this objective on exactly this base model (pi0.5), and
    what the whole I-JEPA/V-JEPA family uses in direct-regression form.

    `context_tokens` / `target_tokens_raw` are Pi0.embed_prefix's raw (batch, 256, 2048)
    right-wrist output -- the former from the online model, the latter from
    `nnx.merge(state.model_def, state.ema_params)`, i.e. already detached from the diffed argnum by
    construction (ema_params is never part of `nnx.value_and_grad`'s differentiated state).
    `target_tokens_raw` is additionally wrapped in `jax.lax.stop_gradient` below as defensive,
    BYOL-standard belt-and-suspenders.

    This REPLACED a flow-matching formulation (noise the target, regress a velocity), which was
    ported from the offline Phase 0 probe on the theory that it would avoid regressing to the mean
    on a multimodal future. Measured on a real 500-step run, that objective was uninterpretable:
    it fell to ~1.03 within 30 steps and sat there, because "estimate the target, treat the noise
    as unpredictable" is an attractor pinned at exactly 1 + s for unexplained target variance s.
    A loss of 1.043 was consistent with s ~ 0.44 OR s ~ 0.043 depending on whether the predictor
    used the noised input at all -- a 10x ambiguity in the only quantity that mattered. A control
    on synthetic data with a known-predictable future confirmed the failure mode: informative
    context scored 1.664 against 1.651 for context carrying NO information, i.e. the objective was
    not using the context at all while still looking converged. Latent prediction already solves
    the multimodality problem the flow wrapper was added for -- the encoder is free to drop
    unpredictable content -- and nothing here ever samples from the flow, so its machinery bought
    only a pedestal that hid the signal.

    Cosine is scale-free, so unlike the flow objective it cannot be inflated by a drifting
    representation scale, and it reads on an interpretable range: 0 is perfect, ~1.0 is
    uncorrelated.

    Returns (per-example loss, per-example copy baseline), both (batch,)."""
    # TOKEN_DIM/NUM_TOKENS are properties of the checkpoint (gemma_2b's width, SigLIP-224's
    # per-camera token count), not of this module -- a different paligemma_variant or image
    # resolution would otherwise fail deep inside AuxProjection's dot_general or
    # AuxFuturePredictor's query_pos_embed broadcast with a shape error that doesn't name the
    # real cause. `.shape` is static under jax.jit tracing, so this check is free and safe here.
    for name, tokens in (("context_tokens", context_tokens), ("target_tokens_raw", target_tokens_raw)):
        if tokens.shape[-2:] != (NUM_TOKENS, TOKEN_DIM):
            raise ValueError(
                f"{name} has shape {tokens.shape}, expected (..., {NUM_TOKENS}, {TOKEN_DIM}) -- "
                "wam_aux.py's TOKEN_DIM/NUM_TOKENS assume gemma_2b + one 224x224 SigLIP camera; "
                "a different paligemma_variant or image resolution needs those constants updated."
            )
    del rng  # direct regression is deterministic given the batch; kept for signature stability

    context = _standardize(projection(context_tokens))
    target = jax.lax.stop_gradient(projection_ema.apply(target_tokens_raw))

    batch = target.shape[0]
    k = jnp.full((batch,), offset_k, dtype=jnp.float32)
    prediction = predictor(context, k=k, actions=action_window)

    per_token = 1.0 - _cosine_similarity(prediction, target)
    # The copy baseline: what "the future just looks like the present" already scores. Logged
    # alongside the loss because the loss alone cannot say whether the predictor is doing anything
    # the identity map would not -- at k=8 (0.27s at 30Hz) the two frames are very close, and a
    # near-identity solution would otherwise look like success.
    copy_baseline = 1.0 - _cosine_similarity(context, target)
    return jnp.mean(per_token, axis=-1), jnp.mean(copy_baseline, axis=-1)
