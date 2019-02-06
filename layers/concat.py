import tensorflow as tf


def build_layer_fn(params):
    name = params.get('__name__')
    fields = params.get('fields')

    def layer_fn(tensor, features):
        tensor_list = []
        for field in fields:
            tensor_list.append(features.get(field))
        concat_tensor = tf.string_join(tensor_list, ',')
        return concat_tensor

    return layer_fn
