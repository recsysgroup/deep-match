import tensorflow as tf

user_id = tf.constant(['1', '2', '3', '3', '5'])

user_embedding = tf.constant([[0.1, 0.15], [0.2, 0.25], [0.3, 0.35], [0.4, 0.45], [0.5, 0.55]])


def unique_user_op(user_id, user_embedding):
    def user_func(user_id, user_embedding):

        unique_user_id = []
        unique_user_embedding = []
        unique_set = set([])
        for i in range(len(user_id)):
            if user_id[i] not in unique_set:
                unique_user_id.append(user_id[i])
                unique_user_embedding.append(user_embedding[i])
                unique_set.add(user_id[i])

        return unique_user_id, unique_user_embedding

    y = tf.py_func(user_func, [user_id, user_embedding], [tf.string, tf.float32])
    y[0].set_shape((None))
    y[1].set_shape((None, user_embedding.get_shape()[1]))
    return y


with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    session.run(tf.tables_initializer())

    print user_id
    print user_embedding

    unique_user_id, unique_user_embedding = unique_user_op(user_id, user_embedding)

    print unique_user_id, unique_user_embedding

    print session.run([unique_user_id, unique_user_embedding])
