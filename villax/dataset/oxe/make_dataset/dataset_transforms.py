"""Open X-Embodiment Dataset Transforms

input: dict of features, each is batched, i.e. has leading time dimension
expected output:
step = {
    'observation': {
        <image_keys, depth_image_keys>
        state in chosen state representation
    },
    'action': action in chosen action representation,
    'language_instruction': str,
}
"""

# to import from another folder.
from typing import Any, Dict

import tensorflow as tf

from .utils.data import (
    binarize_gripper_actions,
    invert_gripper_actions,
    rel2abs_gripper_actions,
    relabel_actions,
)


def bridge_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # NOTE: this is not actually the official OXE copy of bridge, it is our own more up-to-date copy that you
    # can find at https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/

    """
    Applies to original version of Bridge V2 from the official project website.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key == "observation":
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory = relabel_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][
        :, -1:
    ]
    return trajectory


def rt1_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    return trajectory


def libero_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    from ...utils.tf_rotation import axis_angle_to_quaternion

    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            gripper_action,
        ],
        axis=1,
    )
    # process proprio
    state = trajectory["observation"]["state"]
    # 1 if gripper is open, 0 if gripper is closed
    openness = tf.cast(tf.abs(state[:, 6]) + tf.abs(state[:, 7]) > 0.025, tf.float32)
    trajectory["observation"]["state"] = tf.concat(
        [
            state[:, :3],
            axis_angle_to_quaternion(state[:, 3:6]),
            openness[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


OXE_STANDARDIZATION_TRANSFORMS = {
    "bridge_dataset": bridge_dataset_transform,
    "fractal20220817_data": rt1_dataset_transform,
    "libero_10_no_noops": libero_dataset_transform,
    "libero_goal_no_noops": libero_dataset_transform,
    "libero_object_no_noops": libero_dataset_transform,
    "libero_spatial_no_noops": libero_dataset_transform,
}
