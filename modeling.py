import tensorflow as tf
import constant as C
from layers import build_layer_fn


class MatchNet(object):
    def __init__(self, config):
        self.config = config

    def _emb_sum(self, name, features, training):
        emb_list = []
        for fea in self.config.get(name):
            fea_name = fea.get(C.CONFIG_FEATURE_NAME)
            fea_layers = fea.get(C.CONFIG_FEATURE_LAYERS)

            fea_tensor = features.get(fea_name)

            for index, layer in enumerate(fea_layers):
                layer_type = layer.get('type')
                params = {'__name__': fea_name}
                params.update(layer)
                params['__config__'] = self.config
                layer_fn = build_layer_fn(layer_type, params)

                fea_tensor = layer_fn(fea_tensor, features)

            # fea_tensor = tf.layers.dropout(fea_tensor, rate=0.2, training=training)

            emb_list.append(fea_tensor)

        emb_sum = tf.reduce_sum(tf.stack(emb_list, axis=2), axis=-1)

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

    def user_embedding(self, features, training=False):
        return self._emb_sum('user', features, training)

    def item_embedding(self, features, training=False):
        return self._emb_sum('item', features, training)
