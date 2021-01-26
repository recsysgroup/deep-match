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

        user_emb = matchNet.user_embedding(pos_features)
        item_emb = matchNet.item_embedding(pos_features)

        hrs = metric_cal_op(user_emb, item_emb, topk)

        ret = {}
        for index, k in enumerate(topk):
            ret['hr@' + str(k)] = hrs[index]

        return ret

    return eval_fn


def metric_cal_op(user_embedding, item_embedding, topk):
    topk = sorted(topk)
    dis_mat = tf.matmul(user_embedding, item_embedding, transpose_b=True)
    _, index_mat = tf.nn.top_k(dis_mat, max(topk))

    def user_func(_index_mat):

        hit_nums = [0 for _ in range(len(topk))]
        for i in range(len(_index_mat)):
            for j in range(len(_index_mat[i])):
                if _index_mat[i][j] == i:
                    for k in range(len(topk)):
                        if j < topk[k]:
                            hit_nums[k] += 1

        hrs = [hit_num * 1.0 / len(_index_mat) for hit_num in hit_nums]

        return np.array(hrs, dtype=np.float32)

    y = tf.py_func(user_func, [index_mat], tf.float32)
    y.set_shape(len(topk))
    return y
