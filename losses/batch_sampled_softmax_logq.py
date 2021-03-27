import tensorflow as tf
import constant as C
from utils import assign_optimizer


def build_loss_fn(config):
    _loss_params = config.get(C.CONFIG_LOSS)
    l2 = _loss_params.get('l2', 0.0)
    batch_size = config.get('train_batch_size', 128)

    MAX_DELTA = 100
    Q_BUCKET_SIZE = 100000000

    with tf.variable_scope('q', reuse=tf.AUTO_REUSE):
        last_hit_var = tf.get_variable('last_hit',
                                       [Q_BUCKET_SIZE],
                                       initializer=tf.initializers.constant(0))
        last_delta_var = tf.get_variable('last_delta',
                                         [Q_BUCKET_SIZE],
                                         initializer=tf.initializers.constant(MAX_DELTA))
        g1_step = tf.get_variable('g_step', [1],
                                  initializer=tf.initializers.constant(0))

    @tf.custom_gradient
    def assign_add_layer(x):
        def grad(dy):
            return x + 1.0 + 0 * dy

        return tf.identity(x), grad

    @tf.custom_gradient
    def assign_grad_layer(x, y):
        def grad(dy):
            return y, 0 * dy

        return tf.identity(x), grad

    def loss_fn(matchNet, features):
        pos_features = features.get(C.CONFIG_INPUT_POS_FEATURES)
        item = tf.string_to_hash_bucket(pos_features.get('item_id'), Q_BUCKET_SIZE)
        last_hit = tf.nn.embedding_lookup(last_hit_var, item, name='last_hit')
        last_delta = tf.nn.embedding_lookup(last_delta_var, item, name='last_delta')
        g_step_up = assign_add_layer(g1_step)

        new_hit = tf.tile(tf.cast(g1_step, tf.float32), [tf.shape(last_hit)[0]])
        new_delta = tf.maximum(tf.stop_gradient(new_hit - last_hit), 1)
        new_delta = tf.stop_gradient(last_delta * 0.9 + new_delta * 0.1)
        last_delta_up = assign_grad_layer(last_delta, new_delta)
        last_hit_up = assign_grad_layer(last_hit, new_hit)
        item_weight = tf.log(new_delta / MAX_DELTA)

        user_emb = matchNet.user_embedding(pos_features)
        item_emb = matchNet.item_embedding(pos_features)

        all_fm_list = tf.matmul(user_emb, item_emb, transpose_b=True)
        mock_label = tf.eye(tf.shape(all_fm_list)[0])

        loss = tf.losses.softmax_cross_entropy(mock_label, all_fm_list)
        regularizer = tf.contrib.layers.l2_regularizer(l2)
        regloss = tf.contrib.layers.apply_regularization(regularizer)
        tf.summary.histogram("reg_loss", regloss)

        last_hit_up = tf.Print(last_hit_up, [g1_step, last_hit, new_hit, last_delta, new_delta], summarize=1,
                               message='last_hit')

        zero_loss = 0 * tf.reduce_sum(last_hit_up) + 0 * tf.reduce_sum(last_delta_up) + 0 * g_step_up

        learning_rate = tf.constant(value=config["learning_rate"], shape=[], dtype=tf.float32)
        opt = tf.train.AdagradOptimizer(learning_rate)
        fuzhu_opt = assign_optimizer.AssignOptimizer(1.0)
        train_op = opt.minimize(loss, global_step=tf.train.get_global_step())
        fuzhu_op = fuzhu_opt.minimize(zero_loss, global_step=tf.train.get_global_step())

        return loss + regloss + zero_loss, tf.group(train_op, fuzhu_op)

    return loss_fn
