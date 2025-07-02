from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import tensorflow as tf


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()
    }


def tree_merge(*trees: dict) -> dict:
    """Merges a list of nested dictionaries, with later dictionaries overriding earlier ones."""
    merged = {}
    for tree in trees:
        for k, v in tree.items():
            if isinstance(v, dict):
                merged[k] = tree_merge(merged.get(k, {}), v)
            else:
                merged[k] = v
    return merged


def to_padding(tensor: tf.Tensor) -> tf.Tensor:
    if tf.debugging.is_numeric_tensor(tensor):
        return tf.zeros_like(tensor)
    elif tensor.dtype == tf.string:
        return tf.fill(tf.shape(tensor), "")
    else:
        raise ValueError(f"Cannot generate padding for tensor of type {tensor.dtype}.")


def make_neutral_actions(
    action: tf.Tensor, absolute_action_mask: tf.Tensor
) -> tf.Tensor:
    """Returns "neutral" actions, meaning relative actions are zeroed and absolute actions are retained.
    `absolute_action_mask` should be a 1D boolean mask that indicates which action dimensions are absolute.
    """
    return tf.where(
        absolute_action_mask[(None,) * (action.ndim - 1)],
        action,
        tf.zeros_like(action),
    )


def combine_dataset_statistics(
    all_dataset_statistics: Sequence[dict],
) -> dict:
    """Merges dataset statistics from multiple datasets."""
    merge_stat_keys = ["action", "proprio"]

    num_trajectories = [stat["num_trajectories"] for stat in all_dataset_statistics]
    num_transitions = [stat["num_transitions"] for stat in all_dataset_statistics]
    stat_weights = [
        transitions / sum(num_transitions) for transitions in num_transitions
    ]

    combined_dataset_statistics = {}
    for key in merge_stat_keys:
        combined_mean = np.array(
            [
                stat[key]["mean"] * w
                for stat, w in zip(all_dataset_statistics, stat_weights)
            ]
        ).sum(0)
        # compute combined_std for denominator `n` instead of `n-1` since numpy uses that by default for std
        # https://stats.stackexchange.com/questions/55999/is-it-possible-to-find-the-combined-standard-deviation
        combined_std = np.sqrt(
            np.array(
                [
                    n * np.array(stat[key]["std"]) ** 2
                    + n * (np.array(stat[key]["mean"]) - combined_mean) ** 2
                    for stat, n in zip(all_dataset_statistics, num_transitions)
                ]
            ).sum(0)
            / sum(num_transitions)
        )
        combined_dataset_statistics[key] = {
            "min": np.array([stat[key]["min"] for stat in all_dataset_statistics])
            .min(0)
            .tolist(),
            "max": np.array([stat[key]["max"] for stat in all_dataset_statistics])
            .max(0)
            .tolist(),
            "mean": combined_mean.tolist(),
            "std": combined_std.tolist(),
        }

    combined_dataset_statistics["num_trajectories"] = num_trajectories
    combined_dataset_statistics["num_transitions"] = num_transitions
    return combined_dataset_statistics


def binarize_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """Converts gripper actions from continous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near 1.0) or fully closed (near
    0.0). As it transitions between the two, it sometimes passes through a few intermediate values. We relabel
    those intermediate values based on the state that is reached _after_ those intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we give up on binarizing and relabel
    that chunk of intermediate values as the last action in the trajectory.

    The scan implements the following code:

    new_actions = np.empty_like(actions)
    carry = actions[-1]
    for i in reversed(range(actions.shape[0])):
        if in_between_mask[i]:
            carry = carry
        else:
            carry = float(open_mask[i])
        new_actions[i] = carry
    """
    open_mask = actions > 0.95
    closed_mask = actions < 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))

    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(
            in_between_mask[i],
            lambda: tf.cast(carry, tf.float32),
            lambda: is_open_float[i],
        )

    new_actions = tf.scan(
        scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True
    )
    return new_actions


def rel_open_or_closed(actions: tf.Tensor):
    """
    Returns the initial absolute gripper state, given relative actions (-1 for closing, +1 for opening)
    Returns 1 if the gripper is initially open, 0 if it is initially closed.
    If nothing taken, assumes gripper is initially open.

    """
    opening_mask = actions > 1e-3
    closing_mask = actions < -1e-3
    old_state_mask = tf.where(opening_mask, -1, tf.where(closing_mask, -1, 0))
    # old_state_mask is 1 if closing, -1 if opening, 0 if no change

    def scan_fn(carry, i):
        return tf.cond(
            old_state_mask[i] == 0,
            lambda: tf.cast(carry, tf.float32),
            lambda: (tf.cast(old_state_mask[i], tf.float32) + 1) / 2,
        )

    return tf.scan(
        scan_fn,
        tf.range(tf.shape(actions)[0]),
        tf.zeros_like(actions[-1]),
        reverse=True,
    )[0]


def rel2abs_gripper_actions(actions: tf.Tensor):
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute gripper actions
    (0 for closed, 1 for open). Assumes that the first relative gripper is not redundant
    (i.e. close when already closed).
    """
    opening_mask = actions < -0.1
    closing_mask = actions > 0.1

    # -1 for closing, 1 for opening, 0 for no change
    thresholded_actions = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))

    def scan_fn(carry, i):
        return tf.cond(
            thresholded_actions[i] == 0,
            lambda: carry,
            lambda: thresholded_actions[i],
        )

    # if no relative grasp, assumes open for whole trajectory
    start = -1 * thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)
    # -1 for closed, 1 for open
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)

    new_actions = tf.cast(new_actions, tf.float32) / 2 + 0.5
    return new_actions


def invert_gripper_actions(actions: tf.Tensor):
    return 1 - actions


def relabel_actions(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Relabels the actions to use the reached proprio instead. Discards the last timestep of the
    trajectory (since we don't have a next state to compute the action.)
    """
    # relabel the first 6 action dims (xyz position, xyz rotation) using the reached proprio
    movement_actions = (
        traj["observation"]["state"][1:, :6] - traj["observation"]["state"][:-1, :6]
    )

    # discard the last timestep of the trajectory
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)

    # recombine to get full actions
    traj_truncated["action"] = tf.concat(
        [movement_actions, traj["action"][:-1, -1:]],
        axis=1,
    )

    return traj_truncated


def allocate_threads(n: Optional[int], weights: np.ndarray):
    """Allocates an integer number of threads across datasets based on weights. The final array sums to `n`,
    but each element is no less than 1. If `n` is None, then every dataset is assigned a value of AUTOTUNE.
    """
    if n is None:
        return np.array([tf.data.AUTOTUNE] * len(weights))

    n = max(n, len(weights))
    assert np.all(weights >= 0), "Weights must be non-negative"
    assert len(weights) <= n, (
        "Number of threads must be at least as large as length of weights"
    )
    weights = np.array(weights) / np.sum(weights)

    allocation = np.zeros_like(weights, dtype=int)
    while True:
        # give the remaining elements that would get less than 1 a 1
        mask = (weights * n < 1) & (weights > 0)
        if not mask.any():
            break
        n -= mask.sum()
        allocation += mask.astype(int)
        # recompute the distribution over the remaining elements
        weights[mask] = 0
        weights = weights / weights.sum()
    # allocate the remaining elements
    fractional, integral = np.modf(weights * n)
    allocation += integral.astype(int)
    n -= integral.sum()
    for i in np.argsort(fractional)[::-1][: int(n)]:
        allocation[i] += 1
    return allocation
