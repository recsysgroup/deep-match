import tensorflow as tf


def build_aggregator_fn(params):
    l2 = params.get('l2', 0.0)

    def aggregator_fn(tensor_list):
        l2_reg = tf.contrib.layers.l2_regularizer(l2)
        tf.contrib.layers.apply_regularization(l2_reg, tensor_list)
        emb_sum = tf.reduce_sum(tf.stack(tensor_list, axis=2), axis=-1)
        return emb_sum

    return aggregator_fn
