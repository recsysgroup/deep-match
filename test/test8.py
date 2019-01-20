import tensorflow as tf
import numpy as np

user_id = tf.constant(['1', '2', '3', '3', '2'])

pos_dis = tf.constant([0.1, 0.2, 0.3, 0.4, 0.5])

unique_user_id = tf.constant(['1', '2', '3'])

neg_dis = tf.constant([[0.1, 0.11], [0.1, 0.11], [0.1, 0.11]])


def metric_cal_op(user_id, pos_dis, unique_user_id, neg_dis):
    def user_func(user_id, pos_dis, unique_user_id, neg_dis):
        user_id_2_samples = {}


        for i in range(len(unique_user_id)):
            samples = []
            for dis in neg_dis[i]:
                samples.append((dis, 0))

            user_id_2_samples[unique_user_id[i]] = samples

        for i in range(len(user_id)):
            user_id_2_samples[user_id[i]].append((pos_dis[i], 1))

        hit_size = 0
        mrr = 0.0
        for user_id, samples in user_id_2_samples.items():
            samples = sorted(samples, reverse=True)

            for i in range(min(len(samples), 20)):
                if samples[i][1] == 1:
                    hit_size += 1

        hr = hit_size * 1.0 / len(pos_dis)


        return np.float32(hr)


    y = tf.py_func(user_func, [user_id, pos_dis, unique_user_id, neg_dis], [tf.float32])
    return y


with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    session.run(tf.tables_initializer())

    hr = metric_cal_op(user_id, pos_dis, unique_user_id, neg_dis)

    print hr, session.run(hr)
