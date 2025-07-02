from functools import partial
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ...utils.normalize import normalize_state_action
from ..schema import OpenXConfig
from .compute_stats import compute_stats
from .standardize import make_standardized_dataset
from .transforms import obs_transforms, traj_transforms

if TYPE_CHECKING:
    from dlimp import DLataset

    from ..schema import OpenXLoadConfig
tf.config.set_visible_devices([], "GPU")


def apply_per_dataset_frame_transforms(
    dataset: "DLataset",
    chunk_filter_fn: Optional[Callable] = None,
):
    """
    Optionally applied *per-dataset* transforms that happen at a frame level.

    Args:
        chunk_filter_fn (callable, optional): Filter function for chunks.
    """
    if chunk_filter_fn:
        dataset = dataset.filter(chunk_filter_fn)
    return dataset


def apply_trajectory_transforms(
    dataset: "DLataset",
    *,
    sample_length: int = 8,
    sample_interval: int = 3,
    action_offset: int = 0,
    action_chunk: int = 0,
    allow_padding: bool = False,
    padding_side: Literal["left", "right", "both"] = "left",
    min_valid_frames: int = 2,
    min_left_samples: int = 2,
    min_right_samples: int = 2,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> "DLataset":
    if skip_unlabeled:
        if "instruction" not in dataset.element_spec:
            raise ValueError(
                "skip_empty_instruction=True but dataset does not have language labels."
            )
        dataset = dataset.filter(lambda x: tf.math.reduce_any(x["instruction"] != ""))
    if max_action is not None:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(tf.math.abs(x["action"]) <= max_action)
        )

    if max_proprio is not None and "proprio" in dataset.element_spec["observation"]:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(
                tf.math.abs(x["observation"]["proprio"]) <= max_proprio
            )
        )

    min_valid_frames = min(min_valid_frames, sample_length)
    min_valid_frames = max(min_valid_frames, min_left_samples)
    min_valid_frames = max(min_valid_frames, min_right_samples)
    if not allow_padding:
        min_valid_frames = sample_length

    # filter out trajectories that are too short
    dataset = dataset.filter(
        lambda x: tf.math.reduce_all(
            tf.shape(x["action"])[0] >= min_valid_frames * sample_interval
        )
    )

    dataset = dataset.traj_map(
        partial(
            traj_transforms.sample_trajectory,
            sample_length=sample_length,
            sample_interval=sample_interval,
            action_chunk=action_chunk,
            action_offset=action_offset,
            allow_padding=allow_padding,
            padding_side=padding_side,
            min_left_samples=min_left_samples,
            min_right_samples=min_right_samples,
        ),
        num_parallel_calls,
    )

    return dataset


def apply_frame_transforms(
    dataset: "DLataset",
    *,
    is_train: bool,
    image_augment_kwargs: Union[dict, Mapping[str, dict]] = {},
    resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    depth_resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> "DLataset":
    import dlimp as dl

    # convenience wrapper that takes a function that operates on a non-chunked "observation" dict and applies
    # it to the chunked "observation" dict as well as the non-chunked "task" dict
    def apply_obs_transform(fn: Callable[[dict], dict], frame: dict) -> dict:
        frame["observation"] = dl.vmap(fn)(frame["observation"])
        return frame

    # decode + resize images (and depth images)
    dataset = dataset.frame_map(
        partial(
            apply_obs_transform,
            partial(
                obs_transforms.decode_and_resize,
                resize_size=resize_size,
                depth_resize_size=depth_resize_size,
            ),
        ),
        num_parallel_calls,
    )

    if is_train:
        # augment all images with the same seed, skipping padding images
        def aug(frame: dict):
            seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
            aug_fn = partial(
                obs_transforms.augment, seed=seed, augment_kwargs=image_augment_kwargs
            )
            return apply_obs_transform(aug_fn, frame)

        dataset = dataset.frame_map(aug, num_parallel_calls)

    return dataset


def make_interleaved_dataset(
    config: "OpenXLoadConfig",
    dataset_configs: list[OpenXConfig],
    weights: Optional[list[float]] = None,
):
    import dlimp as dl

    # default to uniform sampling
    if not weights:
        weights = [1.0] * len(dataset_configs)
    if len(weights) != len(dataset_configs):
        raise ValueError(
            f"sample_weights must be None or have length {len(dataset_configs)}."
        )

    stats = [
        compute_stats(config, ds_config, force=config.force_compute_stats)
        for ds_config in dataset_configs
    ]
    # balance and normalize weights
    dataset_sizes = [stat.num_transitions for stat in stats]
    if config.balance_weights:
        weights = np.array(weights) * np.array(dataset_sizes)
    weights = np.array(weights) / np.sum(weights)

    if config.threads == -1:
        config.threads = min(config.max_threads, len(dataset_configs))
    else:
        config.threads = min(config.max_threads, config.threads)
    per_ds_threads = np.clip(config.threads * weights, 1, None).astype(int)

    # construct datasets
    datasets = []
    for ds_config, stat, ds_threads in tqdm(
        zip(dataset_configs, stats, per_ds_threads),
        desc="Loading OXE datasets",
        total=len(dataset_configs),
    ):
        dataset = make_standardized_dataset(
            config=ds_config,
            shuffle=ds_config.shuffle,
            is_train=ds_config.is_train,
            threads=ds_threads,
        )
        if ds_config.use_normalization and ds_config.state_obs_keys is not None:
            dataset = dataset.traj_map(
                partial(
                    normalize_state_action,
                    stat=stat,
                    normalization_type=ds_config.normalization_type,
                    action_mask=ds_config.action_normalization_mask,
                    state_mask=ds_config.state_normalization_mask,
                ),
                num_parallel_calls=ds_threads,
            )
        # use repeat to make infinite dataset
        dataset = apply_trajectory_transforms(
            dataset.repeat() if ds_config.infinite_sample else dataset,
            sample_length=ds_config.sample_length,
            sample_interval=ds_config.sample_interval,
            action_offset=ds_config.action_offset,
            action_chunk=ds_config.action_chunk,
            allow_padding=ds_config.allow_padding,
            padding_side=ds_config.padding_side,
            min_valid_frames=ds_config.min_valid_frames,
            min_left_samples=ds_config.min_left_samples,
            min_right_samples=ds_config.min_right_samples,
            skip_unlabeled=ds_config.skip_empty_instruction,
            max_action=ds_config.max_action,
            max_proprio=ds_config.max_proprio,
            num_parallel_calls=ds_threads,
        ).flatten(num_parallel_calls=ds_threads)
        dataset = apply_per_dataset_frame_transforms(dataset, ds_config.chunk_filter_fn)
        datasets.append(dataset)

    # interleave at the frame level and then shuffle
    dataset: "DLataset" = dl.DLataset.sample_from_datasets(datasets, weights)
    if config.shuffle:
        dataset = dataset.shuffle(config.shuffle_buffer_size)

    # apply frame transforms
    dataset = apply_frame_transforms(
        dataset,
        is_train=config.is_train,
        image_augment_kwargs=config.image_augment_kwargs,
        resize_size=config.resolution,
        depth_resize_size=config.resolution,
        num_parallel_calls=config.threads,
    )

    # this seems to reduce memory usage without affecting speed
    dataset = dataset.with_ram_budget(1)

    stats = {ds_config.name: stat for ds_config, stat in zip(dataset_configs, stats)}
    return dataset, weights, stats
