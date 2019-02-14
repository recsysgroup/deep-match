import tensorflow as tf


def build_similarity_fn(params):
    def similarity_fn(user_emb, item_emb):
        pos_dis = tf.reduce_sum(user_emb * item_emb, axis=-1)
        return pos_dis

    return similarity_fn
