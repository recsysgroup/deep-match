import tensorflow as tf
import constant as C


def build_layer_fn(params):
    name = params.get(C.CONFIG_GLOBAL_FEATURE_NAME)
    embedding_dim = params.get('dim')
    stddev = params.get('stddev', 0.01)
    _fea_side = params.get(C.CONFIG_GLOBAL_FEATURE_SIDE)

    if embedding_dim is None:
        config = params.get(C.CONFIG_GLOBAL_CONFIG)
        embedding_dim = config.get('embedding_size')

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        bias = tf.get_variable(name,
                               [1, embedding_dim],
                               initializer=tf.truncated_normal_initializer(stddev=stddev))

    def layer_fn(tensor, features):
        shape = tf.shape(features.get(_fea_side))
        return tf.tile(bias, [shape[0], 1])

    return layer_fn
