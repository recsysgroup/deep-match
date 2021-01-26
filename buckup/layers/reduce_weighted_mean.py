import tensorflow as tf


def build_layer_fn(params):
    weight_name = params.get('weighted_name')

    def layer_fn(tensor, features):
        weight_tensor = features.get(weight_name)
        weight_tensor = tf.expand_dims(tf.string_to_number(weight_tensor, out_type=tf.float32), axis=-1)
        return tf.reduce_sum(tensor * weight_tensor, axis=1) / (tf.reduce_sum(weight_tensor, axis=1) + 1e-6)

    return layer_fn
