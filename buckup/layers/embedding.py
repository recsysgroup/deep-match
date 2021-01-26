import tensorflow as tf
import constant as C


def build_layer_fn(params):
    name = params.get('name')
    embedding_size = params.get('size')
    embedding_dim = params.get('dim')
    stddev = params.get('stddev', 0.01)
    if embedding_dim is None:
        config = params.get(C.CONFIG_GLOBAL_CONFIG)
        embedding_dim = config.get('embedding_size')
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        embedding = tf.get_variable(name,
                                    [embedding_size, embedding_dim],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))

    def layer_fn(tensor, features):
        emb = tf.nn.embedding_lookup(embedding, tensor, name=name + '_lookup')
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, emb)
        return emb

    return layer_fn
