import tensorflow as tf


def build_layer_fn(params):
    name = params.get('name')
    embedding_size = params.get('size')
    embedding_dim = params.get('dim')
    if embedding_dim is None:
        config = params.get('__config__')
        embedding_dim = config.get('embedding_size')
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        embedding = tf.get_variable(name,
                                    [embedding_size, embedding_dim],
                                    initializer=tf.truncated_normal_initializer(stddev=0.01))

    def layer_fn(tensor, features):
        return tf.nn.embedding_lookup(embedding, tensor, name=name + '_lookup')

    return layer_fn
