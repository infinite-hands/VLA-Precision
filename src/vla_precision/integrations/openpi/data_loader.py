"""OpenPI loader construction with all data columns prepared before sampling."""

from __future__ import annotations

from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat

install_lerobot_import_compat()

import logging
import os
from pathlib import Path

import jax
import numpy as np
from openpi import transforms
from openpi.models import model as _model
from openpi.training import data_loader as openpi_data_loader

from vla_precision.config.schema import RootConfig, Stage1Config
from vla_precision.data.indexing import materialize_lerobot_indices
from vla_precision.data.paths import source_lerobot_root
from vla_precision.integrations.openpi.data_configs import SplitAuxFutureFrame
from vla_precision.integrations.openpi.policies.dual_ur import AUX_FUTURE_IMAGE_KEY, AUX_FUTURE_PAD_KEY

logger = logging.getLogger(__name__)


class TransformedDataset:
    """OpenPI-equivalent transform wrapper owned by the spawn-safe extension."""

    def __init__(self, dataset, transform_fns):
        self._dataset = dataset
        self._transform = transforms.compose(transform_fns)

    def __getitem__(self, index):
        return self._transform(self._dataset[index])

    def __len__(self):
        return len(self._dataset)


def _collate_fn(items):
    """Match OpenPI's NumPy collation without importing its loader in workers."""
    return jax.tree.map(
        lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0),
        *items,
    )


def _worker_init_fn(worker_id: int) -> None:
    """Match OpenPI's worker-side JAX allocator setup."""
    del worker_id
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class TorchDataLoader(openpi_data_loader.TorchDataLoader):
    """Keep OpenPI's loader behavior with extension-owned spawn callbacks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_loader.collate_fn = _collate_fn
        self.torch_loader.worker_init_fn = _worker_init_fn


def transform_dataset(dataset, data_config, *, skip_norm_stats: bool = False):
    """Apply OpenPI's transform order with a spawn-safe dataset wrapper."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError("Normalization stats not found. Run Stage-I norm-stats first.")
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def _lerobot_dataset(dataset):
    """Reach the LeRobot owner before OpenPI's optional prompt wrapper."""
    current = dataset
    while not hasattr(current, "hf_dataset") and hasattr(current, "_dataset"):
        current = current._dataset
    return current


def create_torch_dataset(
    data_config,
    action_horizon: int,
    model_config,
    root_config: RootConfig | Stage1Config,
):
    """Mirror OpenPI's LeRobot constructor while honoring the top-level root."""
    aux_enabled = isinstance(root_config, Stage1Config) and root_config.openpi.aux_loss_weight > 0
    if data_config.repo_id == "fake" or root_config.data.lerobot_root is None:
        # See the matching guard in create_data_loader: this branch cannot add the delta_timestamps
        # entry SplitAuxFutureFrame's already-installed repack transform depends on, so it must
        # never run while aux is enabled -- create_data_loader's own guard should already have
        # raised before calling this function with these arguments, but this function is public
        # and callable directly, so it re-checks rather than trusting every caller.
        if aux_enabled:
            raise ValueError(
                "aux_loss_weight > 0 requires a real LeRobot root and repo_id (got "
                f"repo_id={data_config.repo_id!r}, lerobot_root={root_config.data.lerobot_root!r}) -- "
                "this path returns upstream's plain dataset, which cannot supply the extra future "
                "frame SplitAuxFutureFrame expects."
            )
        return openpi_data_loader.create_torch_dataset(
            data_config,
            action_horizon,
            model_config,
        )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    root = source_lerobot_root(root_config)
    metadata = LeRobotDatasetMetadata(data_config.repo_id, root=root)
    delta_timestamps = {
        key: [step / metadata.fps for step in range(action_horizon)]
        for key in data_config.action_sequence_keys
    }
    # WAM auxiliary future-prediction loss (see docs/wam-aux-loss.md): request ONE extra frame,
    # `aux_loss_offset_k` steps ahead, for the right-wrist column only -- not the whole
    # action_horizon schedule above, which would decode 30 future frames per sample instead of 1.
    # Stage-II (RootConfig) never sets this; only Stage1Config.openpi carries it. The raw key is
    # read back from the SplitAuxFutureFrame instance data_configs.py's LeRobotDualUR5eConfig
    # already installed into data_config.repack_transforms (instead of re-deriving it separately
    # from image_key_map here, which would risk the two resolutions silently disagreeing).
    if aux_enabled:
        aux_split = next(
            (t for t in data_config.repack_transforms.inputs if isinstance(t, SplitAuxFutureFrame)),
            None,
        )
        if aux_split is None:
            raise ValueError(
                "aux_loss_weight > 0 but no SplitAuxFutureFrame transform was found in "
                "repack_transforms -- the data config's .create() should have installed one "
                "whenever aux_loss_offset_k is set (see LeRobotDualUR5eDataConfig.create)."
            )
        delta_timestamps[aux_split.raw_key] = [0, root_config.openpi.aux_loss_offset_k / metadata.fps]
    dataset = LeRobotDataset(data_config.repo_id, root=root, delta_timestamps=delta_timestamps)
    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset,
            [transforms.PromptFromLeRobotTask(metadata.tasks)],
        )
    return dataset


class DataLoaderImplWithAux(openpi_data_loader.DataLoaderImpl):
    """WAM auxiliary future-prediction loss (see docs/wam-aux-loss.md): subclasses upstream
    OpenPI's DataLoaderImpl (inheriting its __init__/data_config() unchanged, so an upstream
    signature change surfaces here rather than silently diverging from a hand-copied
    reimplementation) and overrides only __iter__, which pulls the future right-wrist frame OUT of
    the batched `batch["image"]` dict BEFORE calling Observation.from_dict() -- which would
    otherwise either silently drop it or (worse) feed it to embed_prefix() as a fourth real camera,
    since that dataclass builds `images=data["image"]` verbatim -- and yields it as a separate
    third tuple element. It also has to redo, by hand, the one normalization step
    Observation.from_dict() would have applied to it (uint8 [0,255] -> float32 [-1,1]): that step
    iterates `data["image"]`'s keys internally, so removing this key first (required, see above)
    means from_dict() never sees it and never normalizes it. Does not require any upstream OpenPI
    change; train.py's train_step is the only other place that needs to know about the wider
    tuple."""

    def __iter__(self):
        for batch in self._data_loader:
            # No default: this class is only ever constructed when aux is enabled (see
            # create_data_loader below), so a missing key here is always a pipeline defect, never
            # a valid state -- let it raise KeyError at the point of the actual bug instead of
            # silently yielding aux_future_image=None into train_step_with_aux.
            aux_future_image = batch["image"].pop(AUX_FUTURE_IMAGE_KEY)
            if aux_future_image.dtype == np.uint8:
                aux_future_image = aux_future_image.astype(np.float32) / 255.0 * 2.0 - 1.0
            # LeRobot clamps the future index at episode ends, making "future" == current for
            # roughly offset_k/episode_length of samples; those are masked out of the loss.
            aux_is_pad = batch.pop(AUX_FUTURE_PAD_KEY)
            yield _model.Observation.from_dict(batch), batch["actions"], aux_future_image, aux_is_pad


def create_data_loader(
    train_config,
    root_config: RootConfig | Stage1Config,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
):
    """Create the standard OpenPI loader after one eager HF ``map`` pass."""
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    aux_enabled = isinstance(root_config, Stage1Config) and root_config.openpi.aux_loss_weight > 0
    # WAM auxiliary future-prediction loss (see docs/wam-aux-loss.md): "aux is enabled" must be
    # decided in exactly one place and enforced everywhere else, not re-derived independently at
    # each early-return branch below. `LeRobotDualUR5eDataConfig.create()` (data_configs.py)
    # already installed `SplitAuxFutureFrame` into `data_config.repack_transforms` whenever
    # `aux_loss_offset_k is not None`, on the assumption that `create_torch_dataset` below will
    # actually add the matching `delta_timestamps` entry that stacks the right-wrist column into
    # (2, H, W, C) -- but `create_torch_dataset`'s own `fake`/`lerobot_root is None` branch (and
    # this function's `rlds_data_dir`/`fake` branch) skip that entirely and fall back to upstream's
    # plain loader, which would silently hand `SplitAuxFutureFrame` an UNSTACKED (H, W, C) frame:
    # `stacked[0]` then returns a single row of pixels, not a frame -- corrupting the real "now"
    # right-wrist image with no error at the point of the actual bug. Fail loudly here instead,
    # mirroring the existing fail-loud checks in configs.py's build_stage1_train_config.
    if aux_enabled and (data_config.rlds_data_dir is not None or data_config.repo_id == "fake"):
        raise ValueError(
            "aux_loss_weight > 0 requires the LeRobot/torch data path (create_torch_dataset), "
            f"which this data_config cannot use: rlds_data_dir={data_config.rlds_data_dir!r}, "
            f"repo_id={data_config.repo_id!r}."
        )
    if data_config.rlds_data_dir is not None or data_config.repo_id == "fake":
        return openpi_data_loader.create_data_loader(
            train_config,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework="jax",
        )

    dataset = create_torch_dataset(
        data_config,
        train_config.model.action_horizon,
        train_config.model,
        root_config,
    )
    experiment_name = (
        root_config.openpi.exp_name
        if isinstance(root_config, Stage1Config)
        else root_config.experiment.name
    )
    metadata_path = (
        Path(root_config.paths.cache_root)
        / "index_materialization"
        / experiment_name
        / "metadata.json"
    )
    metadata = materialize_lerobot_indices(
        _lerobot_dataset(dataset),
        root_config.data,
        metadata_path=metadata_path,
    )
    logger.info("materialized OpenPI training columns: schema_sha256=%s", metadata.schema_sha256)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    local_batch_size = train_config.batch_size // jax.process_count()
    loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        sampler=None,
        num_batches=num_batches,
        num_workers=train_config.num_workers,
        seed=train_config.seed,
        framework="jax",
    )
    impl = DataLoaderImplWithAux if aux_enabled else openpi_data_loader.DataLoaderImpl
    return impl(data_config, loader)
