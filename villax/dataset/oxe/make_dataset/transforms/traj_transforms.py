from typing import Literal

import tensorflow as tf


def sample_trajectory(
    traj: dict,
    sample_length: int,
    sample_interval: int,
    action_chunk: int,
    action_offset: int = 0,
    allow_padding: bool = True,
    padding_side: Literal["left", "right", "both"] = "left",
    min_left_samples: int = 2,
    min_right_samples: int = 2,
) -> dict:
    n_frames = tf.shape(traj["action"])[0]
    clip_len = sample_length * sample_interval

    # clip indices before padding
    clip_indices = tf.range(0, n_frames)
    """[0 1 2 3 4 5 6 7 8 9] clip_indices"""
    # mask before padding
    mask = tf.ones([n_frames], dtype=tf.bool)

    left_padding = clip_len - min_left_samples * sample_interval
    right_padding = clip_len - 1 - (min_right_samples - 1) * sample_interval

    if allow_padding:
        if padding_side == "left" and left_padding > 0:
            clip_indices = tf.concat(
                [tf.zeros([left_padding], dtype=tf.int32), clip_indices], axis=0
            )
            mask = tf.concat([tf.zeros([left_padding], dtype=tf.bool), mask], axis=0)
        elif padding_side == "right" and right_padding > 0:
            clip_indices = tf.concat(
                [clip_indices, tf.fill([right_padding], n_frames - 1)], axis=0
            )
            mask = tf.concat([mask, tf.zeros([right_padding], dtype=tf.bool)], axis=0)
        elif padding_side == "both":
            if left_padding > 0:
                clip_indices = tf.concat(
                    [tf.zeros([left_padding], dtype=tf.int32), clip_indices], axis=0
                )
                mask = tf.concat(
                    [tf.zeros([left_padding], dtype=tf.bool), mask], axis=0
                )
            if right_padding > 0:
                clip_indices = tf.concat(
                    [clip_indices, tf.fill([right_padding], n_frames - 1)], axis=0
                )
                mask = tf.concat(
                    [mask, tf.zeros([right_padding], dtype=tf.bool)], axis=0
                )

    """[0 0 0 1 2 3 4 5 6 7 8] clip_indices
        [F F T T T T T T T T T] mask
    """

    total_length = tf.shape(clip_indices)[0]
    action_chunk = max(action_chunk, sample_interval)
    n_clips = tf.maximum(total_length - clip_len + 1, 1)
    """[x x x x x x] total_length
        [x x x x] clip_len
    """

    obs_first_indice = tf.range(0, clip_len)[::sample_interval]
    """[0 2 4 6 8] obs_indice"""
    state_gather_indices = tf.range(0, clip_len)
    """[0 1 2 3 ...] state_indice"""

    obs_gather_indices = (
        tf.broadcast_to(obs_first_indice[None, :], [n_clips, sample_length])
        + tf.range(n_clips)[:, None]
    )
    """
    [0 2 4 6 8]
    [1 3 5 7 9]
    ...
    """
    state_gather_indices = (
        obs_gather_indices[:, :, None] + tf.range(0, action_chunk)[None, None, :]
    )
    """
    [[0, 1, 2], [2, 3, 4], [4, 5, 6], ...]
    [[1, 2, 3], [3, 4, 5], [5, 6, 7], ...]
    """
    action_gather_indices = state_gather_indices + action_offset
    state_gather_indices = tf.minimum(
        state_gather_indices, tf.shape(clip_indices)[0] - 1
    )
    action_gather_indices = tf.minimum(
        action_gather_indices, tf.shape(clip_indices)[0] - 1
    )

    obs_clip_indice = tf.gather(clip_indices, obs_gather_indices)
    act_clip_indice = tf.gather(clip_indices, action_gather_indices)
    act_clip_indice = tf.minimum(
        act_clip_indice, n_frames - 1
    )  # ensure but not necessary
    state_clip_indice = tf.gather(clip_indices, state_gather_indices)
    state_clip_indice = tf.minimum(
        state_clip_indice, n_frames - 1
    )  # ensure but not necessary
    actions = tf.gather(traj["action"], act_clip_indice)
    traj["state"] = tf.gather(traj["state"], state_clip_indice)
    traj["observation"] = tf.nest.map_structure(
        lambda x: tf.gather(x, obs_clip_indice), traj["observation"]
    )

    traj["observation"]["pad_mask"] = tf.gather(mask, obs_gather_indices)
    traj["state_mask"] = tf.gather(mask, state_gather_indices)

    for k in ["dataset_name", "control_frequency", "state_per_obs", "has_state"]:
        traj[k] = tf.repeat(traj[k], n_clips)
    # Only keep the first n_clips instructions, TODO may need to change this
    # to support multiple instructions
    traj["instruction"] = tf.gather(traj["instruction"], obs_clip_indice[:, 0])

    # traj["action_indice"] = tf.reshape(act_clip_indice, [n_clips, sample_length, -1])

    if "absolute_action_mask" in traj:
        ashape = actions.shape.as_list()
        ashape[0] = tf.shape(obs_gather_indices)[0]  # Replace None with actual value
        absolute_action_mask = tf.broadcast_to(traj["absolute_action_mask"], ashape)
    else:
        absolute_action_mask = tf.zeros_like(actions, dtype=tf.bool)
    traj["absolute_action_mask"] = absolute_action_mask

    action_mask = tf.gather(mask, action_gather_indices)
    neutral_actions = tf.where(absolute_action_mask, actions, tf.zeros_like(actions))
    traj["action"] = tf.where(action_mask[..., None], actions, neutral_actions)
    return traj
