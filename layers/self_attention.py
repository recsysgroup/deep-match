import tensorflow as tf
import math
import constant as C


def build_layer_fn(params):
    name = params.get(C.CONFIG_GLOBAL_FEATURE_NAME)
    l2 = params.get('l2')

    def layer_fn(tensor, features):
        _, seq_len, dim = tensor.get_shape().as_list()
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2)
            query_layer = tf.layers.dense(tensor, dim, use_bias=False,
                                          kernel_regularizer=regularizer)
            key_layer = tf.layers.dense(tensor, dim, use_bias=False,
                                        kernel_regularizer=regularizer)

        # [B, L, L]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(dim)))

        # [B, L]
        mask0 = features.get(name + '_mask')
        # [B, 1, L]
        mask = tf.cast(tf.reshape(mask0, [-1, 1, seq_len]), tf.float32)

        # [B, L, 1]
        broadcast_ones = tf.transpose(tf.ones_like(mask), [0, 2, 1])

        # [B, L, L]
        attention_mask = broadcast_ones * mask

        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
        attention_scores += adder

        attention_probs = tf.nn.softmax(attention_scores)

        context_layer = tf.matmul(attention_probs, tensor)

        emb = tf.reduce_mean(context_layer, axis=1)

        tf.summary.histogram('self_attention', attention_probs)

        return emb

    return layer_fn
