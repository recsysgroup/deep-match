import tensorflow as tf
import constant as C


def build_policy_fn(config, name):
    fea_conf = config.get(name)
    dnn_conf = fea_conf.get('dnn')
    name2columns = {}
    for column in config.get('columns'):
        name2columns[column.get('name').split(':')[1]] = column

    def policy_fn(features):
        emb_list = []

        for fea in fea_conf.get('features'):
            fea_name = fea.get('name')
            bucket_size = fea.get('bucket_size')
            emb_size = fea.get('embedding_size')
            with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
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
                emb_list.append(emb)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, emb)

        dnn_layer = tf.concat(emb_list, axis=-1)

        with tf.variable_scope('dnn_' + name, reuse=tf.AUTO_REUSE):
            for index, unit in enumerate(dnn_conf):
                if index == len(dnn_conf) - 1:
                    dnn_layer = tf.layers.dense(dnn_layer, unit, activation=None)
                else:
                    dnn_layer = tf.layers.dense(dnn_layer, unit, activation=tf.nn.leaky_relu)

        return dnn_layer

    return policy_fn
