import tensorflow as tf
import constant as C
from layers import build_layer_fn


class MatchNet(object):
    def __init__(self, config):
        self.config = config

    def _emb_sum(self, name, features):
        emb_sum = None
        for fea in self.config.get(name):
            fea_name = fea.get(C.CONFIG_FEATURE_NAME)
            fea_type = fea.get(C.CONFIG_FEATURE_TYPE)
            fea_layers = fea.get(C.CONFIG_FEATURE_LAYERS)

            fea_tensor = features.get(fea_name)

            for index, layer in enumerate(fea_layers):
                layer_type = layer.get('type')
                params = {'__name__': fea_name}
                params.update(layer)
                layer_fn = build_layer_fn(layer_type, params)

                fea_tensor = layer_fn(fea_tensor, features)

            if emb_sum is None:
                emb_sum = fea_tensor
            else:
                emb_sum = emb_sum + fea_tensor

        return emb_sum

    def user_embedding(self, features):
        return self._emb_sum('user', features)

    def item_embedding(self, features):
        return self._emb_sum('item', features)
