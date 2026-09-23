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

    # Which of the three image slots are real (True) vs padding/absent (False), by openpi key.
    # None (the default) marks every slot real, which is IDENTICAL to this class's original
    # behavior for every config that does not set it. Applied to BOTH train and infer, since this
    # transform runs on both -- a single-arm config that only has a meaningful wrist view sets this
    # once and both stages agree, rather than a train-time dataset choice and a separate serve-time
    # one that could silently drift apart.
    active_image_keys: frozenset[str] | None = None

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
        raw_images = {
            "base_0_rgb": _parse_image(data["observation/exterior_image"]),
            "left_wrist_0_rgb": _parse_image(data["observation/left_wrist_image"]),
            "right_wrist_0_rgb": _parse_image(data["observation/right_wrist_image"]),
        }

        def is_active(key: str) -> bool:
            return self.active_image_keys is None or key in self.active_image_keys

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            # Masked-out slots are ALSO zeroed, not just flagged in image_mask below -- belt and
            # braces against a real image leaking signal through some path that does not honor the
            # mask. Pad any non-existent images with zero-arrays of the appropriate shape.
            "image": {
                key: (image if is_active(key) else np.zeros_like(image))
                for key, image in raw_images.items()
            },
            "image_mask": {key: np.bool_(is_active(key)) for key in raw_images},
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
