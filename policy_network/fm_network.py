import tensorflow as tf
import constant as C


def build_policy_fn(config, name):
    fea_conf = config.get(name)
    name2columns = {}
    for column in config.get('columns'):
        name2columns[column.get('name').split(':')[1]] = column

    def policy_fn(features):
        emb_list = []
        weight_list = []

        for fea in fea_conf.get('features').get('order_1'):
            fea_name = fea.get('name')
            bucket_size = fea.get('bucket_size')
            with tf.variable_scope('order_1', reuse=tf.AUTO_REUSE):
                weights = tf.get_variable(fea_name,
                                          [bucket_size, 1],
                                          initializer=tf.constant_initializer(0.1))

                ids = tf.string_to_hash_bucket_fast(features.get(fea_name), bucket_size)
                weight = tf.nn.embedding_lookup(weights, ids, name=fea_name + '_lookup')
                weight_list.append(weight)

        for fea in fea_conf.get('features').get('order_2'):
            fea_name = fea.get('name')
            bucket_size = fea.get('bucket_size')
            emb_size = fea.get('embedding_size')
            with tf.variable_scope('order_2', reuse=tf.AUTO_REUSE):
                embedding = tf.get_variable(fea_name,
                                            [bucket_size, emb_size],
                                            initializer=tf.random_uniform_initializer(-0.1, 0.1))

                ids = tf.string_to_hash_bucket_fast(features.get(fea_name), bucket_size)
                emb = tf.nn.embedding_lookup(embedding, ids, name=fea_name + '_lookup')
                if fea.get('pooling', '') == 'sum_pooling':
                    print (features.get(fea_name + '_mask'))
                    column_info = name2columns.get(fea.get('name'))
                    seq_mask = tf.expand_dims(
                        tf.sequence_mask(features.get(fea_name + '_mask'), column_info.get('seq_len'),
                                         dtype=tf.float32), axis=-1)
                    emb = emb * seq_mask
                    emb = tf.reduce_sum(emb, axis=1)
                emb_list.append(tf.expand_dims(emb, axis=1))
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, emb)

        order_1 = tf.reduce_sum(tf.concat(weight_list, axis=-1), axis=-1, keepdims=True)
        order_2 = tf.reduce_sum(tf.concat(emb_list, axis=1), axis=1)
        shape = tf.shape(order_2)

        if name == 'user':
            pad_vec = tf.tile(tf.expand_dims(tf.constant([1], dtype=tf.float32), axis=0),
                              [shape[0], 1])

            res_vec = tf.concat([order_1, pad_vec, order_2], axis=-1)
        else:
            pad_vec = tf.tile(tf.expand_dims(tf.constant([1], dtype=tf.float32), axis=0),
                              [shape[0], 1])

            sum_square_emb = tf.reduce_sum(tf.reduce_sum(tf.square(tf.concat(emb_list, axis=1)), axis=-1), axis=-1,
                                           keep_dims=True)
            order_2_comb = 0.5 * (tf.reduce_sum(tf.square(order_2), axis=-1, keep_dims=True) - sum_square_emb)
            res_vec = tf.concat([pad_vec, order_1 + order_2_comb, order_2], axis=-1)
        return res_vec

    return policy_fn
