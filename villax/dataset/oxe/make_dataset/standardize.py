from functools import partial
from os.path import join as pjoin
from typing import Literal

import tensorflow as tf
import tensorflow_datasets as tfds
from loguru import logger

from ..schema import GripperEncoding, OpenXConfig


def encode_angle(x: tf.Tensor, encoding: Literal["euler", "quaternion"]) -> tf.Tensor:
    from ...utils.tf_rotation import quaternion_to_euler

    xyz = x[..., :3]
    rot = x[..., 3:]
    gripper = x[..., -1:]

    fn_mapping = {
        ("euler", "euler"): (3, lambda x: x),
        ("quaternion", "euler"): (4, quaternion_to_euler),
    }
    in_dim, fn = fn_mapping[(encoding, "euler")]

    rot = fn(rot[..., :in_dim])
    return tf.concat([xyz, rot, gripper], axis=-1)


def restructure(traj: dict, config: OpenXConfig):
    required_keys = {"observation", "action"}
    if config.language_key:
        required_keys.add(config.language_key)

    # apply a standardization function, if provided
    if config.standardize_fn is not None:
        traj = config.standardize_fn(traj)

    if not all(k in traj for k in required_keys):
        raise ValueError(f"Missing keys: {required_keys - set(traj.keys())}.")

    # extracts images, depth images and proprio from the "observation" dict
    traj_len = tf.shape(traj["action"])[0]
    old_obs, new_obs = traj["observation"], {}
    camera_mask = {}
    if config.image_obs_keys:
        for new, old in config.image_obs_keys.items():
            if old:
                new_obs[f"image_{new}"] = old_obs[old]
                camera_mask[f"image_{new}"] = tf.repeat(True, traj_len)
            else:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)
                camera_mask[f"image_{new}"] = tf.repeat(False, traj_len)
    if config.depth_obs_keys:
        for new, old in config.depth_obs_keys.items():
            if old:
                new_obs[f"depth_{new}"] = old_obs[old]
                camera_mask[f"depth_{new}"] = tf.repeat(True, traj_len)
            else:
                new_obs[f"depth_{new}"] = tf.repeat("", traj_len)
                camera_mask[f"depth_{new}"] = tf.repeat(False, traj_len)

    new_obs["camera_mask"] = camera_mask

    if config.state_obs_keys:
        new_state = encode_angle(
            tf.concat(
                [
                    tf.zeros((traj_len, 1), dtype=tf.float32)  # padding
                    if key is None
                    else tf.cast(old_obs[key], tf.float32)
                    for key in config.state_obs_keys
                ],
                axis=1,
            ),
            config.state_encoding,
        )
        if config.gripper_encoding == GripperEncoding.CLOSENESS:
            # invert gripper state
            gripper = 1 - new_state[:, -1]
            new_state = tf.concat([new_state[:, :-1], gripper[:, None]], axis=-1)
    else:
        new_state = None

    traj["action"] = encode_angle(traj["action"], config.action_encoding)

    # add timestep info
    new_obs["timestep"] = tf.range(traj_len)

    if config.language_key:
        if traj[config.language_key].dtype != tf.string:
            raise ValueError(
                f"Language key {config.language_key} has dtype {traj[config.language_key].dtype}, "
                "but it must be tf.string."
            )
        instruction = traj.pop(config.language_key)

    traj = {
        "observation": new_obs,
        "action": tf.cast(traj["action"], tf.float32),
        "dataset_name": tf.constant(config.name, dtype=tf.string),
    }
    if config.language_key:
        traj["instruction"] = instruction
    if new_state is not None:
        traj["state"] = tf.cast(new_state, tf.float32)
        traj["has_state"] = tf.constant(True, dtype=tf.bool)
    else:
        traj["state"] = tf.zeros((traj_len, 7), dtype=tf.float32)
        traj["has_state"] = tf.constant(False, dtype=tf.bool)
    traj["control_frequency"] = tf.constant(
        config.control_frequency or 0.0, dtype=tf.float32
    )
    traj["state_per_obs"] = tf.constant(config.sample_interval, dtype=tf.float32)

    if config.absolute_action_mask is not None:
        if len(config.absolute_action_mask) != traj["action"].shape[-1]:
            raise ValueError(
                f"Length of absolute_action_mask ({len(config.absolute_action_mask)}) "
                f"does not match action dimension ({traj['action'].shape[-1]})."
            )
        traj["absolute_action_mask"] = tf.convert_to_tensor(
            config.absolute_action_mask, dtype=tf.bool
        )[None]

    return traj


def make_split(
    splits: list[str],
    *,
    split: str | None = None,
    is_train: bool,
    load_ratio: float,
    train_ratio: float,
):
    k2 = int(load_ratio * 100)
    if split is not None:
        if k2 == 100:
            return split
        return f"{split}[:{k2}%]"

    if "val" not in splits:
        k1 = int(load_ratio * train_ratio * 100)
        _split = f"train[:{k1}%]" if is_train else f"train[{k1}%:{k2}%]"
    else:
        _split = f"train[:{k2}%]" if is_train else f"val[:{k2}%]"
    return _split


def make_standardized_dataset(
    config: OpenXConfig,
    shuffle: bool = True,
    split: str | None = None,
    is_train: bool = True,
    threads: int = -1,
):
    import dlimp as dl

    _restructure_fn = partial(restructure, config=config)
    if "libero" in config.name:
        # hack for Libero datasets TODO: fix this in an elgant way
        builder = tfds.builder(config.name, data_dir=config.base_dir)
    else:
        builder = tfds.builder_from_directory(pjoin(config.base_dir, config.name))

    _split = make_split(
        builder.info.splits,
        split=split,
        is_train=is_train,
        load_ratio=config.load_ratio,
        train_ratio=config.train_ratio,
    )
    logger.debug(f"Loading dataset {config.name} with split {_split}.")
    ds = dl.DLataset.from_rlds(
        builder, split=_split, shuffle=shuffle, num_parallel_reads=threads
    )
    ds = ds.traj_map(_restructure_fn, num_parallel_calls=threads)

    logger.debug(
        f"Loaded dataset {config.name} with split {_split}. {ds.cardinality().numpy()} samples."
    )
    return ds
