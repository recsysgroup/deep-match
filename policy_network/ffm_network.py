import tensorflow as tf
import constant as C


def build_policy_fn(config, name):
    fea_conf = config.get(name)
    opposite_fea_conf = config.get('user' if name == 'item' else 'item')
    name2columns = {}
    for column in config.get('columns'):
        name2columns[column.get('name').split(':')[1]] = column

    def policy_fn(features):
        field_pairs = []
        for field in fea_conf.get('fields'):
            for op_field in opposite_fea_conf.get('fields'):
                pair_names = [field.get('field_name'), op_field.get('field_name')]
                pair_names = sorted(pair_names)
                pair_name = '_for_'.join(pair_names)
                field_pairs.append((pair_name, (field, op_field)))

        field_pairs = sorted(field_pairs, key=lambda x: x[0])

        order_2_emb_list = []

        for field_pair in field_pairs:
            field = field_pair[1][0]
            op_field = field_pair[1][1]
            emb_size = min(field.get('embedding_size'), op_field.get('embedding_size'))
            emb_list = []
            for fea in field.get('features'):
                fea_name = fea.get('name')
                bucket_size = fea.get('bucket_size')
                with tf.variable_scope(name + '_order_2', reuse=tf.AUTO_REUSE):
                    embedding = tf.get_variable(field_pair[0] + '_' + fea_name,
                                                [bucket_size, emb_size],
                                                initializer=tf.random_uniform_initializer(-0.1, 0.1))

                    ids = tf.string_to_hash_bucket_fast(features.get(fea_name), bucket_size)
                    emb = tf.nn.embedding_lookup(embedding, ids, name=fea_name + '_lookup')
                    if fea.get('pooling', '') == 'sum_pooling':
                        column_info = name2columns.get(fea.get('name'))
                        seq_mask = tf.expand_dims(
                            tf.sequence_mask(features.get(fea_name + '_mask'), column_info.get('seq_len'),
                                             dtype=tf.float32), axis=-1)
                        emb = emb * seq_mask
                        emb = tf.reduce_sum(emb, axis=1)
                    elif fea.get('pooling', '') == 'avg_pooling':
                        column_info = name2columns.get(fea.get('name'))
                        seq_mask = tf.expand_dims(
                            tf.sequence_mask(features.get(fea_name + '_mask'), column_info.get('seq_len'),
                                             dtype=tf.float32), axis=-1)
                        emb = emb * seq_mask
                        emb = tf.reduce_sum(emb, axis=1) / (tf.reduce_sum(seq_mask, axis=1) + 1e-6)

                    tf.add_to_collection(tf.GraphKeys.WEIGHTS, emb)
                    emb_list.append(tf.expand_dims(emb, axis=1))
            emb_sum = tf.reduce_sum(tf.concat(emb_list, axis=1), axis=1)
            order_2_emb_list.append(emb_sum)

        order_2 = tf.concat(order_2_emb_list, axis=-1)
        shape = tf.shape(order_2)

        # if name == 'user':
        #     pad_vec = tf.tile(tf.expand_dims(tf.constant([1], dtype=tf.float32), axis=0),
        #                       [shape[0], 1])
        #
        #     res_vec = tf.concat([order_1, pad_vec, order_2], axis=-1)
        #
        # else:
        #     with tf.variable_scope('bias', reuse=tf.AUTO_REUSE):
        #         bias = tf.get_variable('item_bias', [1, 1],
        #                                trainable=False,
        #                                initializer=tf.constant_initializer(1.0))
        #         bias_fea = tf.tile(bias, [shape[0], 1])
        #
        #     pad_vec = tf.tile(tf.expand_dims(tf.constant([1], dtype=tf.float32), axis=0),
        #                       [shape[0], 1])
        #
        #     res_vec = tf.concat([bias_fea, pad_vec, order_1, order_2], axis=-1)
        return order_2

    return policy_fn
