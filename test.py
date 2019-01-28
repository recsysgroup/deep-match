import tensorflow as tf
from layers import build_layer_fn

print build_layer_fn('hash', {'size': 5, 'dim': 5})(tf.constant(['33', '22', '44']),{})
print build_layer_fn('embedding', {'name': 'cate_emb', 'size': 5, 'dim': 5})(tf.constant([1, 2, 4]),{})
print build_layer_fn('embedding', {'name': 'cate_emb', 'size': 5, 'dim': 5})(tf.constant([1, 2, 4]),{})
