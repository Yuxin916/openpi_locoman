import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "state": np.random.rand(32),
        "state_mask": np.random.rand(20),
        "images": {
            "main_image": np.random.randint(256, size=(480, 1280, 3), dtype=np.uint8),
            "wrist_image_left": np.random.randint(256, size=(480, 1280, 3), dtype=np.uint8),
            "wrist_image_right": np.random.randint(256, size=(480, 1280, 3), dtype=np.uint8),
        },
        "image_mask": np.random.rand(3),
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
class LocoManInputs(transforms.DataTransformFn):
    # The action dimension of the model.
    # Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0

        # pad the proprioceptive input to the action dimension of the model.
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        base_image = _parse_image(data["observation/main_image"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])

        image_mask = data["observation/image_mask"].numpy()

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image_left,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": wrist_image_right,
            },
            "image_mask": {
                "base_0_rgb": image_mask[0],
                "left_wrist_0_rgb": image_mask[2],
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": image_mask[1],
            },
        }

        actions = transforms.pad_to_dim(data["actions"], self.action_dim)
        inputs["actions"] = actions
        inputs["actions_mask"] = data["actions_mask"].numpy()

        inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LocoManOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :20])}

