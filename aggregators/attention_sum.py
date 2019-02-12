import tensorflow as tf


def build_aggregator_fn(params):
    name = params.get('__name__')
    tensor_name = name + "_agg_attention"
    with tf.variable_scope(tensor_name, reuse=tf.AUTO_REUSE):
        attention_weight = tf.get_variable(name, [64])

    def aggregator_fn(tensor_list):
        # [B, F, E]
        tensors = tf.stack(tensor_list, axis=1)
        # [B, F]
        attention_score = tf.tensordot(tensors, attention_weight, axes=1)
        # [B, F]
        attention_prob = tf.nn.softmax(attention_score)

        tf.summary.histogram(name + '_agg_attention', attention_prob)

        # [B, E] = [B, E, F] * [B, F, 1]
        out = tf.squeeze(tf.matmul(tensors, tf.expand_dims(attention_prob, axis=-1), transpose_a=True))

        return out

    return aggregator_fn
