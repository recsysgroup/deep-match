import tensorflow as tf

def build_aggregator_fn(params):
    name = params.get('__name__')
    l2 = params.get('l2', 0.0)

    def aggregator_fn(tensor_list):
        tensor = tf.concat(tensor_list, axis=1)

        _, input_size = tensor.get_shape().as_list()

        with tf.variable_scope(name + "_aggregator", reuse=tf.AUTO_REUSE):
            hidden = tf.layers.dense(tensor, 64, activation=tf.nn.relu)
            out = tf.layers.dense(hidden, 64)

        return out

    return aggregator_fn
