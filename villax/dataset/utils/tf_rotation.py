import tensorflow as tf
from tensorflow_graphics.geometry.transformation import euler as tf_euler
from tensorflow_graphics.geometry.transformation import quaternion as tf_quat


def axis_angle_to_quaternion(axis_angle: tf.Tensor):
    axis = tf.nn.l2_normalize(axis_angle[..., :3], axis=-1)
    angle = tf.norm(axis_angle, axis=-1, keepdims=True)
    return tf_quat.from_axis_angle(axis, angle)


def quaternion_to_euler(quaternion: tf.Tensor) -> tf.Tensor:
    return tf_euler.from_quaternion(quaternion)
