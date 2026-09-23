"""VLA-Precision robot data factories written in the native OpenPI style."""

from __future__ import annotations

from vla_precision.integrations.openpi.lerobot_compat import install_lerobot_import_compat

install_lerobot_import_compat()

import dataclasses
import pathlib

from openpi import transforms
from openpi.models import model as openpi_model
from openpi.training import config as openpi_config
from typing_extensions import override

from vla_precision.integrations.openpi.policies import dual_ur, franka, ur5e

AUX_FUTURE_RAW_KEY_SUFFIX = "__aux_future"  # appended to the raw LeRobot column name that carries the extra delta_timestamps frame


@dataclasses.dataclass(frozen=True)
class SplitAuxFutureFrame(transforms.DataTransformFn):
    """WAM auxiliary future-prediction loss (see docs/wam-aux-loss.md): the raw LeRobot item's
    `raw_key` column holds a STACKED (2, H, W, C) array when data_loader.py's delta_timestamps
    requests [0, offset_k/fps] for it, instead of the usual single (H, W, C) frame. Runs FIRST,
    before RepackTransform, so it can restore `raw_key` to just the current frame (index 0) --
    every downstream transform for that key is then bit-identical to a run with aux loss
    disabled -- and expose the future frame (index 1) under its own new key, which
    RepackTransform's (separately extended) mapping then forwards into DualURInputs."""

    raw_key: str

    def __call__(self, data: dict) -> dict:
        stacked = data[self.raw_key]
        data[self.raw_key] = stacked[0]
        data[f"{self.raw_key}{AUX_FUTURE_RAW_KEY_SUFFIX}"] = stacked[1]
        return data


def make_robot_data_config_template(factory, *, dual: bool = False):
    """Create the robot-specific OpenPI DataConfig template selected by TrainConfig."""
    return factory(
        repo_id="",
        base_config=openpi_config.DataConfig(
            prompt_from_task=False,
            action_sequence_keys=("action",),
        ),
        image_key_map=(
            {
                "base_0_rgb": "observation.images.exterior_image",
                "left_wrist_0_rgb": "observation.images.left_wrist_image",
                "right_wrist_0_rgb": "observation.images.right_wrist_image",
            }
            if dual
            else {
                "base_0_rgb": "observation.images.exterior_image",
                "left_wrist_0_rgb": "observation.images.wrist_image",
            }
        ),
        extra_delta_transform=False,
    )


@dataclasses.dataclass(frozen=True)
class LeRobotUR5eDataConfig(openpi_config.DataConfigFactory):
    extra_delta_transform: bool = True
    state_key: str = "observation.state"
    action_key: str = "action"
    image_key_map: dict[str, str] | None = None

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: openpi_model.BaseModelConfig,
    ) -> openpi_config.DataConfig:
        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/image": (self.image_key_map or {}).get(
                            "base_0_rgb", "observation.images.exterior_image"
                        ),
                        "observation/wrist_image": (self.image_key_map or {}).get(
                            "left_wrist_0_rgb", "observation.images.wrist_image"
                        ),
                        "observation/state": self.state_key,
                        "actions": self.action_key,
                        "prompt": "task",
                    }
                )
            ]
        )
        data_transforms = transforms.Group(
            inputs=[ur5e.UR5eInputs(model_type=model_config.model_type)],
            outputs=[ur5e.UR5eOutputs()],
        )
        if self.extra_delta_transform:
            mask = transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(mask)],
                outputs=[transforms.AbsoluteActions(mask)],
            )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=openpi_config.ModelTransformFactory()(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDualUR5eDataConfig(openpi_config.DataConfigFactory):
    extra_delta_transform: bool = True
    state_key: str = "observation.state"
    action_key: str = "action"
    image_key_map: dict[str, str] | None = None
    # WAM auxiliary future-prediction loss (see docs/wam-aux-loss.md). None = disabled, and the
    # repack/data_loader pipeline is then byte-identical to a build without this field at all.
    aux_loss_offset_k: int | None = None

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: openpi_model.BaseModelConfig,
    ) -> openpi_config.DataConfig:
        right_wrist_raw_key = (self.image_key_map or {}).get(
            "right_wrist_0_rgb", "observation.images.right_wrist_image"
        )
        repack_structure = {
            "observation/exterior_image": (self.image_key_map or {}).get(
                "base_0_rgb", "observation.images.exterior_image"
            ),
            "observation/left_wrist_image": (self.image_key_map or {}).get(
                "left_wrist_0_rgb", "observation.images.left_wrist_image"
            ),
            "observation/right_wrist_image": right_wrist_raw_key,
            "observation/state": self.state_key,
            "actions": self.action_key,
            "prompt": "task",
        }
        repack_inputs = []
        if self.aux_loss_offset_k is not None:
            repack_inputs.append(SplitAuxFutureFrame(raw_key=right_wrist_raw_key))
            repack_structure[dual_ur.FUTURE_RIGHT_WRIST_REPACK_KEY] = (
                f"{right_wrist_raw_key}{AUX_FUTURE_RAW_KEY_SUFFIX}"
            )
        repack_inputs.append(transforms.RepackTransform(repack_structure))
        repack = transforms.Group(inputs=repack_inputs)
        data_transforms = transforms.Group(
            inputs=[dual_ur.DualURInputs(model_type=model_config.model_type)],
            outputs=[dual_ur.DualUROutputs()],
        )
        if self.extra_delta_transform:
            mask = transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(mask)],
                outputs=[transforms.AbsoluteActions(mask)],
            )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=openpi_config.ModelTransformFactory()(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaDataConfig(openpi_config.DataConfigFactory):
    extra_delta_transform: bool = True
    state_key: str = "observation.state"
    action_key: str = "action"
    image_key_map: dict[str, str] | None = None

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: openpi_model.BaseModelConfig,
    ) -> openpi_config.DataConfig:
        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/image": (self.image_key_map or {}).get(
                            "base_0_rgb", "observation.images.exterior_image"
                        ),
                        "observation/wrist_image": (self.image_key_map or {}).get(
                            "left_wrist_0_rgb", "observation.images.wrist_image"
                        ),
                        "observation/state": self.state_key,
                        "actions": self.action_key,
                        "prompt": "task",
                    }
                )
            ]
        )
        data_transforms = transforms.Group(
            inputs=[franka.FrankaInputs(model_type=model_config.model_type)],
            outputs=[franka.FrankaOutputs()],
        )
        if self.extra_delta_transform:
            mask = transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(mask)],
                outputs=[transforms.AbsoluteActions(mask)],
            )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=openpi_config.ModelTransformFactory()(model_config),
        )
