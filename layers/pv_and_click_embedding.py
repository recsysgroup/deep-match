import tensorflow as tf


def build_layer_fn(params):
    name = params.get('__name__')
    _min = params.get('min') + 0.0
    _max = params.get('max') + 0.0
    _size = params.get('size')
    _dim = params.get('dim')
    _emb_size = _size + 4

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        embedding = tf.get_variable(name,
                                    [_emb_size, _dim],
                                    initializer=tf.truncated_normal_initializer(stddev=0.01))

    def layer_fn(tensor, features):
        tensor = tf.string_to_number(tensor, out_type=tf.float32)
        tensor = tf.minimum(tensor, tf.constant(_max - 1e-6))
        tensor = tf.maximum(tensor, tf.constant(_min))
        tensor = tf.floordiv(tensor - tf.constant(_min), tf.constant((_max - _min) / _size))
        tensor = tf.cast(tensor, dtype=tf.int32) + tf.constant(2)
        tensor = tf.expand_dims(tensor, axis=1)
        tensor = tf.concat([tensor - tf.constant(2), tensor - tf.constant(1), tensor, tensor + tf.constant(1),
                            tensor + tf.constant(2)], 1)

        tensor = tf.nn.embedding_lookup(embedding, tensor, name=name + '_lookup')
        tensor = tf.transpose(tensor, perm=[0, 2, 1])
        tensor = tf.reduce_mean(tensor, axis=-1)

        return tensor

    return layer_fn
