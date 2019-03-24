import tensorflow as tf
import constant as C


def build_aggregator_fn(params):
    opposite_size = params.get('opposite_size')
    components = params.get('components', ['interact', 'linear', 'constant'])
    _side = params.get(C.CONFIG_GLOBAL_FEATURE_SIDE)

    with tf.variable_scope('one_con', reuse=tf.AUTO_REUSE):
        one_con = tf.get_variable(_side, shape=[1, opposite_size], trainable=False,
                                  initializer=tf.ones_initializer())

    def aggregator_fn(tensor_list):
        vector_list = []
        for compo in components:
            if compo == 'interact':
                if 'linear' in components:
                    two_order = tensor_list[1:]
                    emb_sum = tf.reduce_sum(tf.stack(two_order, axis=2), axis=-1)
                    vector_list.append(emb_sum)
                else:
                    emb_sum = tf.reduce_sum(tf.stack(tensor_list, axis=2), axis=-1)
                    vector_list.append(emb_sum)

            if compo == 'linear':
                one_order = tensor_list[0]
                vector_list.append(one_order)

            if compo == 'constant':
                shape = tf.shape(tensor_list[0])
                constant = tf.tile(one_con, [shape[0], 1])
                vector_list.append(constant)

        ret = tf.concat(vector_list, axis=1)

        return ret

    return aggregator_fn
