import tensorflow as tf


def build_similarity_fn(params):
    def similarity_fn(user_emb, item_emb):
        dis = tf.sqrt(tf.reduce_sum(tf.multiply(user_emb - item_emb, user_emb - item_emb), axis=-1))
        return -dis

    return similarity_fn
