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

        xui = matchNet.similarity(user_emb, item_emb)
        xuj = matchNet.similarity(user_emb, neg_item_emb)
        tf.summary.histogram("bpr_pos_score", xui)
        tf.summary.histogram("bpr_neg_score", xuj)
        xuij = xui - xuj

        l2_norm = tf.add_n([
            regulation_rate * tf.reduce_sum(tf.multiply(user_emb, user_emb)),
            regulation_rate * tf.reduce_sum(tf.multiply(item_emb, item_emb)),
            regulation_rate * tf.reduce_sum(tf.multiply(neg_item_emb, neg_item_emb))
        ])

        bprloss = l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(xuij)))  # BPR loss
        bprloss += tf.losses.get_regularization_loss()

        return bprloss

    return loss_fn
