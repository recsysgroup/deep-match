import tensorflow as tf


def build_aggregator_fn(params):
    name = params.get('__name__')
    l2 = params.get('l2', 0.0)
    regularizer = tf.contrib.layers.l2_regularizer(scale=l2)

    def aggregator_fn(tensor_list):
        tensor = tf.concat(tensor_list, axis=1)

        _, input_size = tensor.get_shape().as_list()

        with tf.variable_scope(name + "_aggregator", reuse=tf.AUTO_REUSE):
            hidden1 = tf.layers.dense(tensor, 256, activation=tf.nn.relu, kernel_regularizer=regularizer)
            hidden2 = tf.layers.dense(hidden1, 128, activation=tf.nn.relu, kernel_regularizer=regularizer)
            out = tf.layers.dense(hidden2, 64, kernel_regularizer=regularizer)

        return out

    return aggregator_fn
