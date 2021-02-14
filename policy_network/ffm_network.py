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

        weight_list = []

        for fea in fea_conf.get('order_1'):
            fea_name = fea.get('name')
            bucket_size = fea.get('bucket_size')
            with tf.variable_scope('order_1', reuse=tf.AUTO_REUSE):
                weights = tf.get_variable(fea_name,
                                          [bucket_size, 1],
                                          initializer=tf.constant_initializer(0.1))

                ids = tf.string_to_hash_bucket_fast(features.get(fea_name), bucket_size)
                weight = tf.nn.embedding_lookup(weights, ids, name=fea_name + '_lookup')
                weight_list.append(weight)

        for field in fea_conf.get('fields'):
            field_name = field.get('field_name')
            emb_size = field.get('embedding_size')
            fea_num = len(field.get('features'))
            if fea_num > 1:
                in_field_emb_list = []
                for fea in field.get('features'):
                    fea_name = fea.get('name')
                    bucket_size = fea.get('bucket_size')
                    with tf.variable_scope(name + '_in_filed', reuse=tf.AUTO_REUSE):
                        embedding = tf.get_variable(field_name + '_' + fea_name,
                                                    [bucket_size, emb_size],
                                                    initializer=tf.random_uniform_initializer(-0.1, 0.1))

                        ids = tf.string_to_hash_bucket_fast(features.get(fea_name), bucket_size)
                        emb = tf.nn.embedding_lookup(embedding, ids, name=fea_name + '_lookup')
                        in_field_emb_list.append(tf.expand_dims(emb, axis=1))

                sum_then_square_emb = tf.reduce_sum(tf.square(
                    tf.reduce_sum(tf.concat(in_field_emb_list, axis=1), axis=1)
                ), axis=-1, keep_dims=True)
                square_then_sum_emb = tf.reduce_sum(
                    tf.reduce_sum(tf.square(tf.concat(in_field_emb_list, axis=1)), axis=-1), axis=-1,
                    keep_dims=True)
                in_field_comb_weight = 0.5 * (sum_then_square_emb - square_then_sum_emb)
                weight_list.append(in_field_comb_weight)

        one_weight = tf.reduce_sum(tf.concat(weight_list, axis=-1), axis=-1, keepdims=True)
        shape = tf.shape(order_2)

        if name == 'user':
            pad_vec = tf.tile(tf.expand_dims(tf.constant([1], dtype=tf.float32), axis=0),
                              [shape[0], 1])

            res_vec = tf.concat([one_weight, pad_vec, order_2], axis=-1)

        else:
            pad_vec = tf.tile(tf.expand_dims(tf.constant([1], dtype=tf.float32), axis=0),
                              [shape[0], 1])

            res_vec = tf.concat([pad_vec, one_weight, order_2], axis=-1)
        return res_vec

    return policy_fn
