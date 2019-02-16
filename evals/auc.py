import tensorflow as tf
import constant as C


def build_eval_fn(params):
    def eval_fn(matchNet, features):
        rank_features = features.get(C.CONFIG_INPUT_POINT_FEATURES)
        labels = rank_features[C.CONFIG_INPUT_LABEL]

        user_emb = matchNet.user_embedding(rank_features)
        content_emb = matchNet.item_embedding(rank_features)

        predictions = matchNet.similarity(user_emb, content_emb)
        _auc_metric = tf.metrics.auc(labels, tf.sigmoid(predictions))
        return _auc_metric

    return eval_fn
