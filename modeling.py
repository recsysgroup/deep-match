import tensorflow as tf
import constant as C
from policy_network import build_policy_fn


class MatchNet(object):
    def __init__(self, config, training):
        self.config = config
        self.training = training

    def _emb_sum(self, name, features):
        conf = self.config.get(name)
        policy_network_name = conf.get('policy_network')
        policy_fn = build_policy_fn(policy_network_name, self.config, name)
        emb = policy_fn(features)
        return emb

    def user_embedding(self, features):
        return self._emb_sum('user', features)

    def item_embedding(self, features):
        return self._emb_sum('item', features)
