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

        neg_item_emb_list = []
        neg_size = 10
        for i in range(neg_size):
            start = batch_size / (neg_size + 1) * i
            tmp_emb1 = tf.slice(item_emb, [start, 0], [batch_size - start, -1])
            tmp_emb2 = tf.slice(item_emb, [0, 0], [start, -1])
            tmp_emb_cc = tf.concat([tmp_emb1, tmp_emb2], axis=0)
            neg_item_emb_list.append(tf.expand_dims(tmp_emb_cc, axis=1))

        neg_item_embs = tf.concat(neg_item_emb_list, axis=1)
        neg_inner_product = tf.reduce_sum(tf.multiply(tf.expand_dims(user_emb, axis=1), neg_item_embs), axis=-1)
        pos_inner_product = tf.reduce_sum(tf.multiply(user_emb, item_emb), axis=-1)
        inner_product = tf.concat([tf.expand_dims(pos_inner_product, axis=1), neg_inner_product], -1)
        prob = tf.nn.softmax(inner_product, axis=-1)
        loss = - tf.reduce_mean(tf.log(tf.slice(prob, [0, 0], [-1, 1])))

        regularizer = tf.contrib.layers.l2_regularizer(l2)
        regloss = tf.contrib.layers.apply_regularization(regularizer)
        tf.summary.histogram("reg_loss", regloss)

        learning_rate = tf.constant(value=config["learning_rate"], shape=[], dtype=tf.float32)
        opt = tf.train.AdagradOptimizer(learning_rate)
        train_op = opt.minimize(loss, global_step=tf.train.get_global_step())

        return loss + regloss, train_op

    return loss_fn
