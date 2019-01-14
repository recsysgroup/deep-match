import tensorflow as tf
import numpy as np

arr = '1,2,3'


# arr2 = tf.constant(np.array(arr))
# t3 = tf.string_split(arr2, ',')
#
# t4 = tf.sparse_to_dense(t3.indices, [t3.dense_shape[0], 5], t3.values, default_value='')



def user_define_op(text, max_seq_len):
    def user_func(text):
        ids = []
        mask = []
        parts = text.split(',')
        for i in range(min(len(parts), max_seq_len)) :
            ids.append(parts[i])
            mask.append(1.0)

        ids = ids + [''] * (max_seq_len - len(ids))
        mask = mask + [0.0] * (max_seq_len - len(mask))

        return ids, mask

    y = tf.py_func(user_func, [text], [tf.string, tf.double])
    return y



with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    session.run(tf.tables_initializer())

    # print session.run(arr2)
    #
    # print session.run(t3)
    #
    # print session.run(t4)

    print session.run(user_define_op(arr, 5))

    # print session.run(tensor_sparse)

    # print session.run(tensor_dense)
