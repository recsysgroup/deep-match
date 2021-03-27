import tensorflow as tf
import constant as C


def build_loss_fn(config):
    _loss_params = config.get(C.CONFIG_LOSS)
    l2 = _loss_params.get('l2', 0.0)
    batch_size = config.get('train_batch_size', 128)

    def loss_fn(matchNet, features):
        pos_features = features.get(C.CONFIG_INPUT_POS_FEATURES)

        user_emb = matchNet.user_embedding(pos_features)
        item_emb = matchNet.item_embedding(pos_features)

        all_fm_list = tf.matmul(user_emb, item_emb, transpose_b=True)
        mock_label = tf.eye(tf.shape(all_fm_list)[0])

        loss = tf.losses.softmax_cross_entropy(mock_label, all_fm_list)
        regularizer = tf.contrib.layers.l2_regularizer(l2)
        regloss = tf.contrib.layers.apply_regularization(regularizer)
        tf.summary.histogram("reg_loss", regloss)

        learning_rate = tf.constant(value=config["learning_rate"], shape=[], dtype=tf.float32)
        opt = tf.train.AdagradOptimizer(learning_rate)
        train_op = opt.minimize(loss, global_step=tf.train.get_global_step())

        return loss + regloss, train_op

    return loss_fn
