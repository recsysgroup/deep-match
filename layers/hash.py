import tensorflow as tf


def build_layer_fn(params):
    hash_size = params.get('size')

    def layer_fn(tensor, features):
        return tf.string_to_hash_bucket_fast(tensor, hash_size)

    return layer_fn
