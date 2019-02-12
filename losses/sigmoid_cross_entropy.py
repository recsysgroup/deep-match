import tensorflow as tf
import constant as C


def build_loss_fn(params):
    regulation_rate = params.get('l2', 0.0)

    def loss_fn(matchNet, features):
        rank_features = features.get(C.CONFIG_INPUT_POINT_FEATURES)
        labels = rank_features[C.CONFIG_INPUT_LABEL]

        user_emb = matchNet.user_embedding(rank_features)
        item_emb = matchNet.item_embedding(rank_features)

        predictions = tf.reduce_sum(user_emb * item_emb, axis=1)
        l2_norm = tf.add_n([
            regulation_rate * tf.reduce_sum(tf.multiply(user_emb, user_emb)),
            regulation_rate * tf.reduce_sum(tf.multiply(item_emb, item_emb))
        ])
        rank_loss = l2_norm + tf.losses.sigmoid_cross_entropy(labels, predictions)
        rank_loss += tf.losses.get_regularization_loss()

        return rank_loss

    return loss_fn
