import tensorflow as tf
import constant as C


def build_loss_fn(params):
    regulation_rate = params.get('l2', 0.0)

    def loss_fn(matchNet, features):
        pos_features = features.get(C.CONFIG_INPUT_POS_FEATURES)
        neg_features = features.get(C.CONFIG_INPUT_NEG_FEATURES)

        user_emb = matchNet.user_embedding(pos_features)
        item_emb = matchNet.item_embedding(pos_features)
        neg_item_emb = matchNet.item_embedding(neg_features)

        pos_dis = tf.reduce_sum(user_emb * item_emb, axis=-1)
        neg_dis = tf.matmul(user_emb, neg_item_emb, transpose_b=True)
        predictions = tf.concat([pos_dis, tf.reshape(neg_dis, [-1])], 0)

        l2_norm = tf.add_n([
            regulation_rate * tf.reduce_sum(tf.multiply(user_emb, user_emb)),
            regulation_rate * tf.reduce_sum(tf.multiply(item_emb, item_emb)),
            regulation_rate * tf.reduce_sum(tf.multiply(neg_item_emb, neg_item_emb))
        ])

        tf.summary.histogram('user_emb', user_emb)
        tf.summary.histogram('item_emb', item_emb)

        labels = tf.concat(
            [tf.ones_like(pos_dis), tf.reshape(tf.zeros_like(neg_dis), [-1])], 0)

        nce_loss = l2_norm + tf.losses.sigmoid_cross_entropy(labels, predictions)
        nce_loss += tf.losses.get_regularization_loss()

        return nce_loss

    return loss_fn
