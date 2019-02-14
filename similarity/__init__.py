import importlib


def build_similarity_fn(type, params):
    if type is None:
        type = 'dot_product'
    module_instance = importlib.import_module('similarity.' + type)
    methodToCall = getattr(module_instance, 'build_similarity_fn')
    return methodToCall(params)
