import tensorflow as tf
import constant as C


def build_loss_fn(params):
    def loss_fn(matchNet, features):
        rank_features = features.get(C.CONFIG_INPUT_POINT_FEATURES)
        labels = rank_features[C.CONFIG_INPUT_LABEL]

        user_emb = matchNet.user_embedding(rank_features)
        item_emb = matchNet.item_embedding(rank_features)

        predictions = tf.reduce_sum(user_emb * item_emb, axis=1)
        rank_loss = tf.losses.sigmoid_cross_entropy(labels, predictions)
        rank_loss += tf.losses.get_regularization_loss()

        return rank_loss

    return loss_fn
