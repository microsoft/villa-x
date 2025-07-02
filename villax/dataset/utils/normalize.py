from typing import TYPE_CHECKING, Optional

import tensorflow as tf

from ..oxe.schema import NormalizationType

if TYPE_CHECKING:
    from ..oxe.schema import OXEDatasetStatistics


def normalize_state_action(
    traj: dict,
    stat: "OXEDatasetStatistics",
    normalization_type: NormalizationType,
    action_mask: Optional[tf.Tensor] = None,
    state_mask: Optional[tf.Tensor] = None,
):
    masks = {"action": action_mask, "state": state_mask}
    metadata = {"action": stat.action, "state": stat.state}

    for k, mask in masks.items():
        if k not in traj:
            continue
        mask = (
            tf.cast(mask, dtype=tf.bool)
            if mask is not None
            else tf.ones_like(metadata[k].mean, dtype=tf.bool)
        )

        if normalization_type == NormalizationType.NORMAL:
            mean = tf.cast(metadata[k].mean, dtype=tf.float32)
            std = tf.cast(metadata[k].std, dtype=tf.float32)

            traj[k] = tf.where(mask, (traj[k] - mean) / (std + 1e-8), traj[k])
        elif normalization_type == NormalizationType.BOUNDS:
            min_val = tf.cast(metadata[k].min, dtype=tf.float32)
            max_val = tf.cast(metadata[k].max, dtype=tf.float32)

            traj[k] = tf.where(
                mask,
                tf.clip_by_value(
                    2 * (traj[k] - min_val) / (max_val - min_val + 1e-8) - 1, -1, 1
                ),
                traj[k],
            )
        elif normalization_type == NormalizationType.QUANTILE:
            q01 = tf.cast(metadata[k].q01, dtype=tf.float32)
            q99 = tf.cast(metadata[k].q99, dtype=tf.float32)
            traj[k] = tf.where(
                mask,
                tf.clip_by_value(2 * (traj[k] - q01) / (q99 - q01 + 1e-8) - 1, -1, 1),
                traj[k],
            )
        else:
            raise ValueError(f"Unknown normalization type {normalization_type}")

    return traj
