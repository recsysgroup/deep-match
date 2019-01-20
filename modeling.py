import tensorflow as tf

class MatchNet(object):
    def __init__(self, config):
        self.config = config

        embedding_dim = 64

        embedding_dict = {}
        for fea in config.get('user') + config.get('item'):
            fea_name = fea.get('name')
            fea_size = fea.get('size')
            fea_type = fea.get('type')
            if fea_type in ('one_hot', 'seq'):
                embedding_dict[fea.get('name')] = tf.get_variable(
                    fea_name + '_embedding',
                    [fea_size, embedding_dim],
                    initializer=tf.truncated_normal_initializer(stddev=0.01))

        self.embedding_dict = embedding_dict

    def _emb_sum(self, name, features):
        emb_sum = None
        for fea in self.config.get(name):
            fea_name = fea.get('name')
            fea_size = fea.get('size')
            fea_type = fea.get('type')

            emb = None
            if fea_type == 'one_hot':
                hash = tf.string_to_hash_bucket_fast(features.get(fea_name), fea_size, name=fea_name + '_hash')
                emb = tf.nn.embedding_lookup(self.embedding_dict.get(fea_name), hash)

            if fea_type == 'seq':
                hash = tf.string_to_hash_bucket_fast(features.get(fea_name), fea_size, name=fea_name + '_hash')
                mask = tf.expand_dims(tf.cast(features.get(fea_name + '_mask'), dtype=tf.float32), axis=-1)
                emb = tf.nn.embedding_lookup(self.embedding_dict.get(fea_name), hash) * mask
                emb = tf.reduce_sum(emb, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-6)

            if emb_sum is None:
                emb_sum = emb
            else:
                emb_sum = emb_sum + emb

        return emb_sum

    def user_embedding(self, features):
        return self._emb_sum('user', features)

    def item_embedding(self, features):
        return self._emb_sum('item', features)
