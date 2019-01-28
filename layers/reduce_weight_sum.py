import tensorflow as tf


def build_layer_fn(params):
    weight_name = params.get('weight_name')

    def layer_fn(tensor, features):
        return tf.reduce_sum(tensor * weight_name, axis=1) / (tf.reduce_sum(weight_name, axis=1) + 1e-6)

    return layer_fn
