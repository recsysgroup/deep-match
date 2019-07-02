import tensorflow as tf
import constant as C


def build_loss_fn(params):
    l2 = params.get('l2', 0.0)

    def loss_fn(matchNet, features):
        pair_features = features.get(C.CONFIG_INPUT_PAIR_FEATURES)
        print pair_features

        user_emb = matchNet.user_embedding(pair_features)
        print '###', user_emb
        item_emb = matchNet.item_embedding(pair_features)
        print '###', item_emb
        neg_item_emb = matchNet.neg_item_embedding(pair_features)
        print '###', neg_item_emb

        xui = matchNet.similarity(user_emb, item_emb)
        xuj = matchNet.similarity(user_emb, neg_item_emb)
        tf.summary.histogram("bpr_pos_score", xui)
        tf.summary.histogram("bpr_neg_score", xuj)
        xuij = xui - xuj

        bprloss = -tf.reduce_mean(tf.log(tf.sigmoid(xuij) + 1e-7))  # BPR loss
        regularizer = tf.contrib.layers.l2_regularizer(l2)
        regloss = tf.contrib.layers.apply_regularization(regularizer)
        tf.summary.histogram("reg_loss", regloss)

        return bprloss + regloss

    return loss_fn
