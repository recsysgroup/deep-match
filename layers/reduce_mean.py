import tensorflow as tf


def build_layer_fn(params):
    name = params.get('name')

    def layer_fn(tensor, features):
        mask = tf.expand_dims(tf.cast(features.get(name + '_mask'), dtype=tf.float32), axis=-1)
        return tf.reduce_sum(tensor * mask, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-6)

    return layer_fn
