from enum import Enum
from typing import Any, Callable, Literal

from pydantic import BaseModel, computed_field

from .make_dataset.dataset_transforms import OXE_STANDARDIZATION_TRANSFORMS

ENC2DIM = {"euler": 6, "quaternion": 7, "rot6d": 9, "axis_angle": 6}


NORMAL_AUGMENT_ARGS = {
    "random_brightness": [0.1],
    "random_contrast": [0.9, 1.1],
    "random_saturation": [0.9, 1.1],
    "random_hue": [0.05],
    "augment_order": [
        "random_brightness",
        "random_contrast",
        "random_saturation",
        "random_hue",
    ],
}

WITH_CROP_AUGMENT_ARGS = {
    **NORMAL_AUGMENT_ARGS,
    "random_resized_crop": {"scale": [0.8, 1.0], "ratio": [0.9, 1.1]},
    "augment_order": [
        "random_resized_crop",
        "random_brightness",
        "random_contrast",
        "random_saturation",
        "random_hue",
    ],
}

IMAGE_AUGMENT_MAPPING = {
    "normal": NORMAL_AUGMENT_ARGS,
    "with_crop": WITH_CROP_AUGMENT_ARGS,
}


def get_mask(encoding: Literal["euler", "quaternion", "rot6d"], invert: bool = False):
    v1, v2 = not invert, invert
    if encoding == "euler":
        return [v1] * 6 + [v2]
    elif encoding == "quaternion":
        return [v1] * 7 + [v2]
    elif encoding == "rot6d":
        return [v1] * 9 + [v2]
    else:
        raise ValueError(f"Unknown state encoding: {encoding}")


class NormalizationType(str, Enum):
    """Defines supported normalization schemes for action and proprio."""

    NORMAL = "normal"  # normalize to mean 0, std 1
    BOUNDS = "bounds"  # normalize to [-1, 1]
    QUANTILE = "quantile"  # normalize to quantiles


class StateEncoding(str, Enum):
    """Defines supported proprio state encoding schemes for different datasets."""

    NONE = "none"  # no state provided
    POS_EULER = "euler"  # EEF XYZ + roll-pitch-yaw + 1 x pad + gripper open/close
    POS_QUAT = "quaternion"  # EEF XYZ + quaternion([x y z w]) + gripper open/close
    JOINT = "joint"  # 7 x joint angles (padding added if fewer) + gripper open/close
    JOINT_BIMANUAL = "joint_bimanual"  # 2 x [6 x joint angles + gripper open/close]


class ActionEncoding(str, Enum):
    """Defines supported action encoding schemes for different datasets."""

    EEF_POS = "euler"  # EEF delta XYZ + roll-pitch-yaw + gripper open/close
    EEF_QUAT = "quaternion"
    JOINT_POS = "joint"  # 7 x joint delta position + gripper open/close
    JOINT_POS_BIMANUAL = "joint_bimanual"  # 2 x [6 x joint pos + gripper]
    EEF_R6 = "rot6d"  # EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1)


class GripperEncoding(str, Enum):
    NONE = "none"
    OPENNESS = "openness"
    CLOSENESS = "closeness"


class ListStatistic(BaseModel):
    mean: list[float]
    std: list[float]
    min: list[float]
    max: list[float]
    q01: list[float]
    q99: list[float]


class ItemStatistic(BaseModel):
    mean: float
    std: float
    min: float | int
    max: float | int
    q01: float | int
    q99: float | int


class EmbodiedDatasetConfig(BaseModel):
    name: str | list[tuple[str, float]]
    base_dir: str
    is_train: bool = True
    load_ratio: float = 1.0  # loaded data / total data
    resolution: tuple[int, int] = (256, 256)
    sample_length: int = 8
    sample_interval: int | float = 3
    allow_padding: bool = True
    min_valid_frames: int = 2
    min_left_samples: int | None = None
    min_right_samples: int | None = None
    padding_side: Literal["left", "right", "both"] = "left"
    action_chunk: int = 0
    action_chunk_interval: int = 1
    shuffle: bool = True

    infinite_sample: bool = True
    force_compute_stats: bool = False
    load_camera_views: list[str] = ["primary"]

    state_dim: int = 7
    action_dim: int = 7
    gripper_dim: int = 1

    use_normalization: bool = False
    normalization_type: str = "normal"

    effective_size: bool = True

    def model_post_init(self, context):
        if self.min_left_samples is None:
            self.min_left_samples = self.min_valid_frames
        if self.min_right_samples is None:
            self.min_right_samples = self.min_valid_frames

    @classmethod
    def default_value(cls, key: str) -> object:
        return cls.model_fields[key].default


class DatasetStatistics(BaseModel):
    config: EmbodiedDatasetConfig | None = None
    num_transitions: int = 0
    num_trajectories: int = 0
    length: ItemStatistic | None = None
    action: ListStatistic | None = None
    state: ListStatistic | None = None


class OpenXLoadConfig(EmbodiedDatasetConfig):
    name: str | list[tuple[str, float]]

    action_offset: int = 0
    max_action: float | None = None
    max_proprio: float | None = None
    skip_empty_instruction: bool = False

    train_ratio: float = 0.95  # train data / loaded data
    shuffle_buffer_size: int = 10000

    image_augment_strategy: Literal["normal", "with_crop"] | None = "normal"
    image_augment_kwargs: dict | None = None

    threads: int = -1
    gripper_dim: int = 1
    max_threads: int = 16
    balance_weights: bool = True
    load_depth: bool = False
    load_proprio: bool = True
    load_language: bool = True

    def model_post_init(self, context):
        if isinstance(self.name, str):
            from .mixture import OXE_NAMED_MIXES

            if self.name in OXE_NAMED_MIXES:
                self.name = OXE_NAMED_MIXES[self.name]

        image_aug_kwargs = {}
        if self.image_augment_strategy:
            assert self.image_augment_strategy in IMAGE_AUGMENT_MAPPING, (
                f"{self.image_augment_strategy} is not an allowed augement strategy."
            )
            f"Avaiable options: {list(IMAGE_AUGMENT_MAPPING.keys())}"
            image_aug_kwargs = {
                k: IMAGE_AUGMENT_MAPPING[self.image_augment_strategy]
                for k in self.load_camera_views
            }
        self.image_augment_kwargs = image_aug_kwargs
        super().model_post_init(context)


class OpenXConfig(OpenXLoadConfig):
    # predefined
    image_obs_keys: dict
    depth_obs_keys: dict | None
    state_encoding: StateEncoding
    action_encoding: ActionEncoding

    # set later
    name: str

    language_key: str | None = "language_instruction"
    standardize_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    state_obs_keys: list[str | None] | None = None
    chunk_filter_fn: Callable | None = None
    gripper_encoding: GripperEncoding = GripperEncoding.OPENNESS
    control_frequency: float | None = None
    gripper_dim: int = 1

    @computed_field
    @property
    def absolute_action_mask(self) -> list[bool]:
        return [False] * 6 + [True]

    @computed_field
    @property
    def action_normalization_mask(self) -> list[bool]:
        return [True] * 6 + [False]

    @computed_field
    @property
    def state_normalization_mask(self) -> list[bool]:
        return [True] * 7

    def model_post_init(self, context):
        if isinstance(self.sample_interval, float):
            if self.control_frequency is None:
                raise ValueError(
                    "control_frequency must be set when sample_interval is a float."
                )
            self.sample_interval = max(
                1, int(self.control_frequency * self.sample_interval)
            )
        self.image_obs_keys = {
            k: self.image_obs_keys.get(k, None) for k in self.load_camera_views
        }
        if self.load_depth:
            self.depth_obs_keys = {
                k: self.depth_obs_keys.get(k, None) for k in self.load_camera_views
            }
        else:
            self.depth_obs_keys = None

        if self.state_encoding in [
            StateEncoding.JOINT,
            StateEncoding.NONE,
            StateEncoding.JOINT_BIMANUAL,
        ]:
            self.state_obs_keys = None

        if not self.load_language:
            self.language_key = None
        self.standardize_fn = OXE_STANDARDIZATION_TRANSFORMS.get(self.name, None)
        super().model_post_init(context)


class OXEDatasetStatistics(DatasetStatistics):
    config: OpenXConfig | None = None
