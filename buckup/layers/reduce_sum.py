import tensorflow as tf
import constant as C

def build_layer_fn(params):

    name = params.get(C.CONFIG_GLOBAL_FEATURE_NAME)

    def layer_fn(tensor, features):
        mask = tf.expand_dims(tf.cast(features.get(name + '_mask'), dtype=tf.float32), axis=-1)
        return tf.reduce_sum(tensor * mask, axis=1)

    return layer_fn
