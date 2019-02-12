import tensorflow as tf
import math


def build_aggregator_fn(params):
    name = params.get('__name__')
    l2 = params.get('l2', 0.0)

    def aggregator_fn(tensor_list):
        # [B, F, E]
        tensor = tf.stack(tensor_list, axis=1)
        _, field_size, dim = tensor.get_shape().as_list()
        with tf.variable_scope(name + "_aggregator", reuse=tf.AUTO_REUSE):
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2)
            query_layer = tf.layers.dense(tensor, dim, activation=tf.nn.tanh, use_bias=False,
                                          kernel_regularizer=regularizer)
            key_layer = tf.layers.dense(tensor, dim, activation=tf.nn.tanh, use_bias=False,
                                        kernel_regularizer=regularizer)

        # [B, F, F]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(dim)))

        attention_probs = tf.nn.softmax(attention_scores)
        tf.summary.histogram('self_attention', attention_probs)

        context_layer = tf.matmul(attention_probs, tensor)

        emb = tf.reduce_mean(context_layer, axis=1)

        return emb

    return aggregator_fn
