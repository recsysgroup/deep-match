import tensorflow as tf
import constant as C


def build_layer_fn(params):
    name = params.get(C.CONFIG_GLOBAL_FEATURE_NAME)
    stddev = params.get('stddev', 0.01)
    _fea_side = params.get(C.CONFIG_GLOBAL_FEATURE_SIDE)

    need_bias = params.get('need_bias', False)

    _features = params.get('features', [])

    emb_dic = {}

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if need_bias:
            bias = tf.get_variable('bias', shape=[1], initializer=tf.constant_initializer(0.01))

        for fea in _features:
            fea_name = fea.get('name')
            fea_size = fea.get('hash')
            emb_dic[fea_name] = tf.get_variable(fea_name, shape=[fea_size],
                                                initializer=tf.truncated_normal_initializer(stddev=stddev))

    def layer_fn(_, features):

        tensor_list = []

        if need_bias:
            shape = tf.shape(features.get(_fea_side))
            bias_fea = tf.tile(bias, [shape[0]])
            tensor_list.append(bias_fea)

        for fea in _features:
            fea_name = fea.get('name')
            fea_size = fea.get('hash')
            fea_val = features.get(fea_name)
            fea_val = tf.string_to_hash_bucket_fast(fea_val, fea_size)
            fea_val = tf.nn.embedding_lookup(emb_dic[fea_name], fea_val)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, fea_val)
            tensor_list.append(fea_val)

        ret = tf.stack(tensor_list, axis=1)
        return ret

    return layer_fn
