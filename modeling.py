import tensorflow as tf
import constant as C
from layers import build_layer_fn
from aggregators import build_aggregator_fn
from similarity import build_similarity_fn


class MatchNet(object):
    def __init__(self, config, training):
        self.config = config
        self.training = training

    def _emb_sum(self, name, features, prefix=''):
        emb_list = []
        for fea in self.config.get(name):
            fea_name = fea.get(C.CONFIG_FEATURE_NAME)
            fea_layers = fea.get(C.CONFIG_FEATURE_LAYERS)

            fea_tensor = features.get(prefix + fea_name)

            for index, layer in enumerate(fea_layers):
                layer_type = layer.get(C.CONFIG_FEATURE_LAYERS_TYPE)
                params = {
                    C.CONFIG_GLOBAL_FEATURE_SIDE: '__{0}__'.format(name),
                    C.CONFIG_GLOBAL_FEATURE_NAME: fea_name,
                    C.CONFIG_GLOBAL_TRAINING: self.training,
                    C.CONFIG_GLOBAL_CONFIG: self.config
                }
                params.update(layer)
                layer_fn = build_layer_fn(layer_type, params)

                fea_tensor = layer_fn(fea_tensor, features)

            # fea_tensor = tf.layers.dropout(fea_tensor, rate=0.5, training=self.training)

            tf.summary.histogram(fea_name, fea_tensor)
            emb_list.append(fea_tensor)

        _aggregator_params = self.config.get(C.CONFIG_AGGREGATORS, {}).get(name, {})
        _aggregator_params['__name__'] = name
        _aggregator_params[C.CONFIG_GLOBAL_FEATURE_SIDE] = '__{0}__'.format(name)
        _aggregator_type = _aggregator_params.get(C.CONFIG_AGGREGATORS_TYPE)
        aggregator_fn = build_aggregator_fn(_aggregator_type, _aggregator_params)

        emb_sum = aggregator_fn(emb_list)

        return emb_sum

    # def _field_dropout_sum(self, emb_list, keeprate=0.8, training=False):
    #     fiels_size =
    #
    #
    #     noise_shape = tf.nn.array_ops.shape(x)
    #     random_tensor = keeprate + tf.nn.random_ops.random_uniform(noise_shape)
    #
    #     tf.nn.random_ops.random_uniform()
    #     pass

    def user_embedding(self, features):
        return self._emb_sum('user', features)

    def item_embedding(self, features):
        return self._emb_sum('item', features)

    def neg_item_embedding(self, features):
        return self._emb_sum('item', features, 'neg__')

    def similarity(self, user_emb, item_emb):
        tf.summary.histogram('user_emb', user_emb)
        tf.summary.histogram('item_emb', item_emb)
        _similarity_params = self.config.get(C.CONFIG_SIMILARITY, {})
        _similarity_type = _similarity_params.get(C.CONFIG_SIMILARITY_TYPE)
        _similarity_fn = build_similarity_fn(_similarity_type, _similarity_params)
        return _similarity_fn(user_emb, item_emb)

    def layers_embedding(self, features):
        name_2_emb = {}
        for fea in self.config.get('user') + self.config.get('item'):
            fea_name = fea.get(C.CONFIG_FEATURE_NAME)
            fea_layers = fea.get(C.CONFIG_FEATURE_LAYERS)

            fea_tensor = features.get(fea_name)

            for index, layer in enumerate(fea_layers):
                layer_type = layer.get(C.CONFIG_FEATURE_LAYERS_TYPE)
                params = {
                    C.CONFIG_GLOBAL_FEATURE_NAME: fea_name,
                    C.CONFIG_GLOBAL_TRAINING: self.training,
                    C.CONFIG_GLOBAL_CONFIG: self.config
                }
                params.update(layer)
                layer_fn = build_layer_fn(layer_type, params)

                fea_tensor = layer_fn(fea_tensor, features)

            name_2_emb[fea_name] = fea_tensor
        return name_2_emb
