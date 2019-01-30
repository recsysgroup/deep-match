import tensorflow as tf


def build_layer_fn(params):
    name = params.get('__name__')
    embedding_size = params.get('size')
    embedding_dim = params.get('dim')
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        embedding = tf.get_variable(name,
                                    [embedding_size, embedding_dim],
                                    initializer=tf.truncated_normal_initializer(stddev=0.01))

    def layer_fn(tensor, features):
        print embedding
        return tf.nn.embedding_lookup(embedding, tensor, name=name + '_lookup')

    return layer_fn
