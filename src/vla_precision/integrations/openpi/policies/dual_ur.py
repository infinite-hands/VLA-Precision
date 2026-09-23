import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def make_dual_ur_example() -> dict:
    """Creates a random input example for the dual_ur policy."""
    return {
        # left tcp_pose/vel/force/torque/gripper, followed by the same
        # fields for the right arm: 19 + 19 dimensions.
        "observation/state": np.random.rand(38),
        "observation/exterior_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


# The one camera the WAM auxiliary future-prediction loss targets (see docs/wam-aux-loss.md).
# Named here since this is where the real "image" dict keys are defined; train.py imports this
# rather than re-declaring the literal.
RIGHT_WRIST_CAMERA_KEY = "right_wrist_0_rgb"

# The repack-level wire key data_configs.py's LeRobotDualUR5eDataConfig writes the future
# right-wrist frame under, and this file's DualURInputs reads it back from (see
# docs/wam-aux-loss.md). Named once here rather than as a bare string literal in both files, so a
# typo on either side cannot silently degrade to "aux quietly disabled" (DualURInputs below guards
# with `if ... in data`, so a mismatched key would otherwise just skip the aux branch, not error).
FUTURE_RIGHT_WRIST_REPACK_KEY = "observation/future_right_wrist_image"

# WAM auxiliary future-prediction loss (see docs/wam-aux-loss.md): the future right-wrist frame
# is stashed under this key INSIDE inputs["image"] (not as its own top-level key) purely so
# ResizeImages -- which resizes every key in data["image"] generically -- resizes it exactly like
# the three real cameras. It carries no "image_mask" entry and DataLoaderImplWithAux
# (data_loader.py) pops it back out of the batched "image" dict before Observation.from_dict()
# runs, since from_dict() builds `images=data["image"]` verbatim and would otherwise hand it to
# embed_prefix() as a fourth real camera.
AUX_FUTURE_IMAGE_KEY = f"__aux_future_{RIGHT_WRIST_CAMERA_KEY}"


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DualURInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        exterior_image = _parse_image(data["observation/exterior_image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": exterior_image,
                "left_wrist_0_rgb": left_wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                # This is a real camera in dual-arm mode, never padding.
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # WAM auxiliary future-prediction loss (see docs/wam-aux-loss.md), only present when
        # aux_loss_weight > 0. Placed inside inputs["image"] under AUX_FUTURE_IMAGE_KEY (see its
        # docstring above) so ResizeImages resizes it identically to the real cameras;
        # DataLoaderImplWithAux strips it back out of "image" before Observation.from_dict() so it
        # never reaches embed_prefix()'s per-camera loop.
        if FUTURE_RIGHT_WRIST_REPACK_KEY in data:
            inputs["image"][AUX_FUTURE_IMAGE_KEY] = _parse_image(data[FUTURE_RIGHT_WRIST_REPACK_KEY])

        return inputs


@dataclasses.dataclass(frozen=True)
class DualUROutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # Dual UR5e uses left 6D+gripper followed by right 6D+gripper.
        return {"actions": np.asarray(data["actions"][:, :14])}
