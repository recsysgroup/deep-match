import tensorflow as tf
import constant as C


def build_layer_fn(params):
    name = params.get(C.CONFIG_GLOBAL_FEATURE_NAME)
    _min = params.get('min') + 0.0
    _max = params.get('max') + 0.0
    _size = params.get('size')
    _dim = params.get('dim')

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        embedding = tf.get_variable(name,
                                    [_size, _dim],
                                    initializer=tf.truncated_normal_initializer(stddev=0.01))

    def layer_fn(tensor, features):
        tensor = tf.string_to_number(tensor, out_type=tf.float32)
        tensor = tf.minimum(tensor, tf.constant(_max - 1e-6))
        tensor = tf.maximum(tensor, tf.constant(_min))
        tensor = tf.floordiv(tensor - tf.constant(_min), tf.constant((_max - _min) / _size))
        tensor = tf.cast(tensor, dtype=tf.int32)
        return tf.nn.embedding_lookup(embedding, tensor, name=name + '_lookup')

    return layer_fn
