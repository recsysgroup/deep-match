import tensorflow as tf


def build_aggregator_fn(params):
    def aggregator_fn(tensor_list):
        emb_sum = tf.reduce_sum(tf.stack(tensor_list, axis=2), axis=-1)
        return emb_sum

    return aggregator_fn
