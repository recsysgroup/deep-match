import tensorflow as tf

def build_layer_fn(params):
    def layer_fn(tensor, features):
        return tf.reduce_sum(tensor, axis=1)

    return layer_fn
