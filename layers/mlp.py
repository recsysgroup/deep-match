import tensorflow as tf
import constant as C
import os


def build_layer_fn(params):
    name = params.get('name')
    dims = params.get('dims')
    _training = params.get(C.CONFIG_GLOBAL_TRAINING)
    _drop_rate = params.get('drop_rate', 0.0)

    def layer_fn(tensor, features):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            initializer = tf.truncated_normal_initializer(0.0, 1e-5)
            regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5)
            for index, dim in enumerate(dims):
                _activation = None
                if index < len(dims) - 1:
                    _activation = tf.nn.relu

                tensor = tf.layers.dropout(tensor, _drop_rate, training=_training,
                                           name=name + '_dropout_' + str(index))

                tensor = tf.layers.dense(tensor, dim,
                                         kernel_initializer=initializer,
                                         activation=_activation,
                                         # kernel_regularizer=regularizer,
                                         name=name + '_dense_' + str(index))

                with tf.variable_scope(name + '_dense_' + str(index), reuse=True):
                    weights = tf.get_variable('kernel')
                    tf.summary.histogram('mlp_weight', weights)

                tf.summary.histogram('mlp_out', tensor)

        return tensor

    return layer_fn
