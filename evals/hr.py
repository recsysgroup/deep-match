import tensorflow as tf
from helper import unique_user_op
import numpy as np
import constant as C


def build_eval_fn(params):
    topk = params.get('topk')
    if type(topk) is not list:
        topk = [topk]

    def eval_fn(matchNet, features):
        pos_features = features.get(C.CONFIG_INPUT_POS_FEATURES)

        user_id = pos_features.get(C.CONFIG_INPUT_USER)
        item_id = pos_features.get(C.CONFIG_INPUT_ITEM)
        user_emb = matchNet.user_embedding(pos_features)
        item_emb = matchNet.item_embedding(pos_features)

        neg_features = features.get(C.CONFIG_INPUT_NEG_FEATURES)
        neg_item_id = neg_features.get(C.CONFIG_INPUT_ITEM)
        neg_item_emb = matchNet.item_embedding(features.get(C.CONFIG_INPUT_NEG_FEATURES))

        hrs = metric_cal_op(user_id, user_emb, item_id, item_emb, neg_item_id, neg_item_emb, topk)

        ret = {}
        for index, k in enumerate(topk):
            ret['hr@' + str(k)] = hrs[index]

        return ret

    return eval_fn


def metric_cal_op(user_id, user_embedding, item_id, item_embedding, neg_item_id, neg_item_embedding, topk):
    pos_dis = tf.reduce_sum(user_embedding * item_embedding, axis=-1)
    unique_user_id, unique_user_embedding = unique_user_op(user_id, user_embedding)
    neg_dis = tf.matmul(unique_user_embedding, neg_item_embedding, transpose_b=True)
    neg_dis, _ = tf.nn.top_k(neg_dis, max(topk))

    def user_func(user_id, item_id, pos_dis, unique_user_id, neg_item_id, neg_dis):
        user_id_2_samples = {}

        for i in range(len(unique_user_id)):
            samples = []
            for j, dis in enumerate(neg_dis[i]):
                samples.append((dis, 0, neg_item_id[j]))

            user_id_2_samples[unique_user_id[i]] = samples

        for i in range(len(user_id)):
            user_id_2_samples[user_id[i]].append((pos_dis[i], 1, item_id[i]))

        hit_nums = [0 for _ in topk]
        for user_id, samples in user_id_2_samples.items():
            samples = sorted(samples, key=lambda x: (x[0], x[1]), reverse=True)
            unique_samples = []
            item_set = set([])
            for i, t in enumerate(samples[0:max(topk)]):
                if t[2] not in item_set:
                    unique_samples.append(t)
                    item_set.add(t[2])

            for i in range(len(unique_samples)):
                if unique_samples[i][1] == 1:
                    for index, k in enumerate(topk):
                        if i < k:
                            hit_nums[index] += 1

        hrs = [hit_num * 1.0 / len(pos_dis) for hit_num in hit_nums]

        return np.array(hrs, dtype=np.float32)

    y = tf.py_func(user_func, [user_id, item_id, pos_dis, unique_user_id, neg_item_id, neg_dis], tf.float32)
    y.set_shape(len(topk))
    return y
