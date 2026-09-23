from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat

install_lerobot_import_compat()

import dataclasses
import functools
import logging
import platform
from typing import Any

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import optax
import tqdm_loggable.auto as tqdm
from etils import epath
from flax import nnx, traverse_util
from flax.training import common_utils
from openpi.shared import nnx_utils
from openpi.training import sharding

import wandb
from vla_precision.config import ResolvedStage1Config
from vla_precision.integrations.openpi import data_loader as _data_loader
from vla_precision.integrations.openpi import wam_aux
from vla_precision.integrations.openpi.configs import build_stage1_train_config
from vla_precision.integrations.openpi.policies.dual_ur import RIGHT_WRIST_CAMERA_KEY as _AUX_CAMERA_KEY

LOGGER = logging.getLogger(__name__)


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
    root_config: dict[str, Any] | None = None,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config={
                "openpi": dataclasses.asdict(config),
                "vla-precision": root_config or {},
            },
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def _apply_model_update(
    config: _config.TrainConfig,
    model: _model.BaseModel,
    state: training_utils.TrainState,
    grads: nnx.State,
    loss: at.Array,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """The optimizer-step + EMA-update + info-dict tail shared by `train_step` and
    `train_step_with_aux`. Extracted so the two step functions cannot silently drift apart --
    before this, the tail was duplicated line-for-line, and the aux path's own docstring claim of
    being "identical to train_step except..." had nothing enforcing it."""
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    return _apply_model_update(config, model, state, grads, loss)


@at.typecheck
def train_step_with_aux(
    config: _config.TrainConfig,
    aux_loss_weight: float,
    aux_loss_offset_k: int,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    aux_state: wam_aux.AuxState,
    batch: tuple[_model.Observation, _model.Actions, at.Float[at.Array, "*b h w c"]],
) -> tuple[training_utils.TrainState, wam_aux.AuxState, dict[str, at.Array]]:
    """WAM auxiliary future-prediction loss (see docs/wam-aux-loss.md): identical to `train_step`
    except it also differentiates a combined (primary + aux_loss_weight * aux) loss w.r.t. the aux
    predictor's own parameters, and steps those via their own separate optimizer (`wam_aux.
    apply_aux_update`) instead of growing Pi0's own trainable_filter. Only called when
    `config.openpi.aux_loss_weight > 0`; `train_step` above is untouched and remains exactly what
    every non-aux run uses, so this function existing changes nothing about stock training."""
    model = nnx.merge(state.model_def, state.params)
    model.train()
    projection = nnx.merge(aux_state.projection_graphdef, aux_state.projection_params)
    predictor = nnx.merge(aux_state.predictor_graphdef, aux_state.predictor_params)

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        projection: wam_aux.AuxProjection,
        predictor: wam_aux.AuxFuturePredictor,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        aux_future_image: at.Array,
    ):
        primary_rng, aux_rng = jax.random.split(rng)
        primary_loss = jnp.mean(model.compute_loss(primary_rng, observation, actions, train=True))

        # Minimal single-camera Observations -- embed_prefix() never reads `.state` (confirmed
        # against the pinned openpi source), so the SAME real `observation.state` is reused for
        # both; only the image differs. This keeps embed_prefix()'s per-camera loop from ever
        # seeing more than the one camera this loss targets, so it cannot leak the future frame
        # into the model's actual perception of "now" via the primary loss's own forward pass
        # above (which used the original, unmodified `observation`).
        now_obs = _model.Observation(
            images={_AUX_CAMERA_KEY: observation.images[_AUX_CAMERA_KEY]},
            image_masks={_AUX_CAMERA_KEY: observation.image_masks[_AUX_CAMERA_KEY]},
            state=observation.state,
        )
        future_obs = _model.Observation(
            images={_AUX_CAMERA_KEY: aux_future_image},
            image_masks={_AUX_CAMERA_KEY: observation.image_masks[_AUX_CAMERA_KEY]},
            state=observation.state,
        )
        context_tokens, _, _ = model.embed_prefix(now_obs)
        # openpi's own existing checkpoint-quality EMA, reused as the BYOL/JEPA-style target
        # encoder: `state.ema_params` is never part of this function's diffed argnums, so the
        # future frame's embedding is detached from the gradient tape by construction (and
        # wrapped in an explicit jax.lax.stop_gradient a second time inside compute_aux_loss,
        # BYOL-standard belt-and-suspenders).
        target_model = nnx.merge(state.model_def, state.ema_params)
        target_tokens_raw, _, _ = target_model.embed_prefix(future_obs)

        per_example_aux, per_example_copy = wam_aux.compute_aux_loss(
            projection,
            predictor,
            aux_state.projection_ema,
            aux_rng,
            context_tokens=context_tokens,
            target_tokens_raw=target_tokens_raw,
            offset_k=aux_loss_offset_k,
            action_window=actions[:, :aux_loss_offset_k, :],
        )
        aux_loss = jnp.mean(per_example_aux)
        total_loss = primary_loss + aux_loss_weight * aux_loss
        return total_loss, (primary_loss, aux_loss, jnp.mean(per_example_copy))

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, aux_future_image = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    argnums = (diff_state, nnx.DiffState(1, nnx.All(nnx.Param)), nnx.DiffState(2, nnx.All(nnx.Param)))
    (loss, (primary_loss, aux_loss, aux_copy_baseline)), (grads, grads_projection, grads_predictor) = (
        nnx.value_and_grad(loss_fn, argnums=argnums, has_aux=True)(
            model, projection, predictor, train_rng, observation, actions, aux_future_image
        )
    )

    new_state, info = _apply_model_update(config, model, state, grads, loss)
    new_aux_state = wam_aux.apply_aux_update(aux_state, grads_projection, grads_predictor)
    info["primary_loss"] = primary_loss
    info["aux_loss"] = aux_loss
    # What "the future looks just like the present" already scores. aux_loss meaningfully below
    # this is the only evidence the predictor is doing something an identity map would not.
    info["aux_copy_baseline"] = aux_copy_baseline
    return new_state, new_aux_state, info


def _train(config: _config.TrainConfig, resolved: ResolvedStage1Config):
    LOGGER.info("running on %s", platform.node())

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(
        config,
        resuming=resuming,
        enabled=config.wandb_enabled,
        root_config=dataclasses.asdict(resolved.config),
    )

    data_loader = _data_loader.create_data_loader(
        config,
        resolved.config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    LOGGER.info("initialized data loader:\n%s", training_utils.array_tree_to_info(batch))

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    LOGGER.info(
        "initialized train state:\n%s",
        training_utils.array_tree_to_info(train_state.params),
    )

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # WAM auxiliary future-prediction loss (see docs/wam-aux-loss.md): decided once, statically,
    # before any jit tracing -- everything below this point either builds the exact stock jit'd
    # step (aux disabled, byte-identical to a build of this file without wam_aux.py at all) or the
    # aux-aware one, never a mix. aux_state is NOT threaded through checkpoint save/restore yet
    # (a known, deliberate gap -- see docs/wam-aux-loss.md): resuming a run restarts the aux
    # predictor and its optimizer from scratch even though `train_state` itself resumes correctly.
    aux_enabled = resolved.config.openpi.aux_loss_weight > 0
    if aux_enabled:
        aux_state = wam_aux.init_aux_state(
            action_dim=config.model.action_dim,
            learning_rate=resolved.config.openpi.aux_learning_rate,
            rngs=nnx.Rngs(jax.random.fold_in(init_rng, 0x4157)),
        )
        ptrain_step_aux = jax.jit(
            functools.partial(
                train_step_with_aux,
                config,
                resolved.config.openpi.aux_loss_weight,
                resolved.config.openpi.aux_loss_offset_k,
            ),
            in_shardings=(replicated_sharding, train_state_sharding, replicated_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding, replicated_sharding),
            donate_argnums=(1, 2),
        )
    else:
        aux_state = None
        ptrain_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            if aux_enabled:
                train_state, aux_state, info = ptrain_step_aux(train_rng, train_state, aux_state, batch)
            else:
                train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    LOGGER.info("waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


def run_stage1_training(resolved: ResolvedStage1Config) -> None:
    """Run standard OpenPI full-parameter training from the unified config."""
    train_config = build_stage1_train_config(resolved.config)
    _train(train_config, resolved)
