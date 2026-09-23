# WAM-style auxiliary future-prediction loss

An optional, training-time-only auxiliary loss for Stage-I: alongside the normal flow-matching
action loss, the model also learns to predict its own future right-wrist embedding from its
current one. Disabled by default (`Stage1OpenPIConfig.aux_loss_weight = 0.0`); when disabled,
every code path this feature touches is byte-identical to a build without it at all.

## Motivation

Stage-I trains purely on (vision, state) → action supervision. Nothing in that loss asks the
vision backbone to represent what the scene is going to do next, so its representations are only
as good as they need to be to reproduce training actions, not to understand geometry or dynamics.
An offline probe (`feat/wam-embedding-probe` in the main `infinite-hands` repo,
`aux_probe_predictor.py`) confirmed real, exploitable future-predictive structure already exists
in the frozen Stage-I checkpoint's right-wrist embedding, and that this structure is present in
the *base* (never fine-tuned) checkpoint too — fine-tuning doesn't erode it. The value proposition
here is therefore not "prevent erosion," but "make the deployed policy actually use predictive
structure that's already latent in its own representations."

## What predicts what

- **Context** (`z_t`): `Pi0.embed_prefix()`'s right-wrist token grid (256 × 2048), computed from
  the **online** (currently-training) parameters, on the current frame.
- **Target** (`z_{t+k}`): the same call, on the **same future frame** `k` steps ahead, but computed
  from `nnx.merge(state.model_def, state.ema_params)` — openpi's own existing checkpoint-quality
  EMA, reused here as the BYOL/JEPA-style target encoder. `state.ema_params` is never part of the
  differentiated argument to `nnx.value_and_grad`, so this is detached from the gradient tape by
  construction; it's additionally wrapped in `jax.lax.stop_gradient` as defensive,
  BYOL-standard belt-and-suspenders.
- No separate bidirectional self-attention "context encoder" is built for this (unlike the offline
  probe, which needed one because it only had frozen, cached embeddings to work with) — SigLIP /
  PaliGemma's vision tower already *is* that encoder, and it's already being trained by the primary
  loss. The one genuinely new, freshly-initialized piece is a small down-projection
  (`token_dim=2048 → d_model=384`, `wam_aux.AuxProjection`), which therefore needs its *own* fresh
  EMA lag (`wam_aux._AuxProjectionEma`, momentum 0.99) to preserve the online/target asymmetry a
  self-predictive loss needs to avoid collapsing to a constant — reusing the backbone's own
  `ema_decay` (tuned for checkpoint-quality EMA, typically 0.999) for this one small new layer
  would not carry the same guarantee.

## Predictor architecture

Ported from the offline probe's validated design (`aux_probe_predictor.py`):

- A **learned positional query** per output token (`query_pos_embed`), added to the current
  flow-matching noise estimate — refined by cross-attention to context, not a content-free mask
  token, since this operates on a noised continuous value (flow-matching), not a masked-prediction
  scheme.
- Per predictor block (`wam_aux._CrossAttnPredictorBlock`): bidirectional self-attention among the
  query tokens, then cross-attention to the context tokens as K/V — both **adaLN-zero** conditioned
  on (flow-matching time `tau`, prediction offset `k`, the action window `t..t+k`). adaLN-zero
  means every block starts as an exact identity map (all gates initialize to zero); gradient only
  starts reaching upstream of the predictor (into the projection / backbone) after the first
  optimizer step turns the gates on — this is expected, not a bug.
- The action window is encoded by a GRU (`wam_aux._ActionSequenceEncoder`) over `actions[:,
  :aux_loss_offset_k, :]` — order-aware, unlike a mean-pool over the window.
- Flow-matching happens entirely in the projection's `d_model` space, not the raw 2048-dim token
  space (cheaper, and matches what was validated offline): uniform `tau ∈ [0, 1]`,
  `noised = (1 - tau) * noise + tau * target`, velocity target `target - noise`. This is ported
  byte-for-byte from the validated offline code, not pi0.5's own Beta(1.5, 1) time-sampling
  convention — the two aren't the same mechanism.
- Both `context` and `target` are standardized per token (`wam_aux._standardize`, a LayerNorm with
  no learnable parameters) before the loss. This is load-bearing, not cosmetic — see below.

## Why the target is normalized (a real bug this caught)

The first 500-step validation run looked healthy for ~40 steps and then went wrong: `aux_loss` fell
16.9 → 2.2, then climbed monotonically back to 15.1 over the remaining ~450 steps, while the
primary model's gradient norm went from ~0.3 to spikes of 4.9-6.9.

The cause was not the prediction task degrading. It was that the loss was a raw, *unnormalized* MSE
in a representation space whose scale nothing constrained. The EMA target's magnitude tracks the
online backbone's, so once the backbone's embedding magnitudes drifted upward during fine-tuning,
the MSE inflated mechanically. Because gradient reaches the backbone through `context`, the loss
was then feeding a growing *scale-driven* rather than *signal-driven* gradient into the model being
fine-tuned — which is what the primary gradient-norm spikes were.

Phase 0's offline probe never hit this because its backbone was frozen: target scale was constant
by construction, and that assumption silently broke once the loss trained jointly with a live
backbone. Isolated reproduction (no backbone, synthetic inputs, 500 steps):

| scenario | climb ratio |
|---|---|
| stationary input scale, unnormalized target | 1.04x (flat) |
| inflating input scale, unnormalized target | 2.31x (reproduces the failure) |
| inflating input scale, normalized target | 1.00x (fixed) |

Normalizing both sides is what BYOL/JEPA implementations do for exactly this reason. It costs the
per-token magnitude as a predictable signal, which is an accepted trade for a bounded objective.

## Separate optimizer state

The predictor + projection have their **own** Adam optimizer (`wam_aux.AuxState`,
`wam_aux.init_aux_state`, `wam_aux.apply_aux_update`), stepped via a **separate**
`nnx.value_and_grad` diffed argument (`argnums=(DiffState(0, trainable_filter), DiffState(1,
All(Param)), DiffState(2, All(Param)))` — one combined loss, three independent gradients, three
independent updates). This keeps the predictor out of Pi0's own `trainable_filter` and optimizer
state entirely, so nothing about `Pi0`'s own NNX class or checkpoint format needs to change.

`AuxState` is a sibling to `training_utils.TrainState`, not merged into it, and is **not** yet
threaded through checkpoint save/restore — resuming a run currently restarts the aux predictor and
its optimizer from scratch even though `train_state` itself resumes correctly. This is a known,
deliberate gap, not an oversight.

## Data plumbing

`Stage1OpenPIConfig.aux_loss_offset_k` (default 8) requests one extra LeRobot `delta_timestamps`
frame for the right-wrist column only (not the whole `action_horizon` schedule, which would decode
many more future frames than needed). `SplitAuxFutureFrame` (`data_configs.py`) splits the
resulting stacked `(2, H, W, C)` array back into "now" (restored in place, so every downstream
transform for that key is bit-identical to aux-disabled) and "future" (exposed under its own repack
key, `dual_ur.FUTURE_RIGHT_WRIST_REPACK_KEY`). `DualURInputs` places the future frame **inside**
`inputs["image"]` under a private key (`dual_ur.AUX_FUTURE_IMAGE_KEY`) so `ResizeImages` resizes it
identically to the three real cameras, then `DataLoaderImplWithAux` (`data_loader.py`) pops it back
out of the batched `"image"` dict and applies by hand the one normalization step
`Observation.from_dict()` would otherwise have applied (uint8 → [-1, 1] float32) — `from_dict()`
never sees this key, so it can't drop it *or* accidentally treat it as a fourth real camera.

"Aux enabled" (`aux_loss_weight > 0`) is decided once, in `create_data_loader`, which fails loudly
if the selected data path (`rlds_data_dir`, a `fake` repo_id, or an unset `lerobot_root`) cannot
actually deliver the stacked future frame `SplitAuxFutureFrame` expects — rather than silently
falling back to upstream's plain 2-tuple loader while the training loop still expects three.

## Known gaps / deliberate simplifications

- No GradNorm-style dynamic loss balancing — `aux_loss_weight` is a fixed scalar. Doing this for
  real would mean a double-backward through the *entire* backbone (not just a small proxy
  encoder), a materially bigger cost/risk than the offline probe's own GradNorm approximation.
- `AuxState` is not part of checkpoint save/restore yet.
- No image augmentation is applied to the aux future frame (only resize + normalize) — this
  matches what the offline probe validated, not `Pi0.compute_loss`'s own internal augmentation
  (which the right-wrist camera mostly skips anyway: only color jitter applies to wrist cameras,
  not the geometric crop/rotate applied to the base camera).
- Calling `embed_prefix()` twice more per step (once for the online context, once for the EMA
  target) is a real, modest added compute cost — two extra vision-tower-only forward passes per
  step, on top of the primary loss's own full forward+backward pass.
