import tensorflow as tf


def build_layer_fn(name, params):
    embedding_dim = params.get('dim')
    pos_name = params.get('pos_name')
    pos_size = params.get('pos_size')
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        pos_embedding = tf.get_variable(pos_name,
                                        [pos_size, embedding_dim],
                                        initializer=tf.truncated_normal_initializer(stddev=0.01))

    def layer_fn(tensor, features):
        pos_emb = tf.nn.embedding_lookup(pos_name, features.get(name))
        return tensor + pos_emb

    return layer_fn
