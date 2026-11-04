"""openpi data transforms for the Kinova Gen3 (6 joints + gripper), modelled on libero_policy / the UR5 example.
Input keys (dataset after repack, and what the 4090 client sends): image, wrist_image (uint8 HxWx3 RGB),
state (7,), prompt (str); actions (H, 7) in training. Needs the openpi venv."""
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 7


def _parse_image(image) -> np.ndarray:
    """LeRobot gives float32 CHW in [0, 1]; the client gives uint8 HWC. Return uint8 HWC."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class KinovaInputs(transforms.DataTransformFn):
    """Map our keys to the pi0/pi0.5 input dict; right wrist slot zero-filled and masked out (pi05 is not FAST)."""

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        base = _parse_image(data["image"])
        wrist = _parse_image(data["wrist_image"])
        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {"base_0_rgb": base, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": np.zeros_like(base)},
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class KinovaOutputs(transforms.DataTransformFn):
    """Model actions (H, 32) -> (H, 7): 6 joint targets (rad, absolute after AbsoluteActions) + gripper 0..1."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}


def make_kinova_example(size: int = 224) -> dict:
    """Random inference-style input (what the 4090 client sends)."""
    return {
        "image": np.random.randint(256, size=(size, size, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(size, size, 3), dtype=np.uint8),
        "state": np.random.uniform(-np.pi, np.pi, 7).astype(np.float32),
        "prompt": "put the foam bee in the tape roll",
    }
