import importlib


def build_layer_fn(type, params):
    module_instance = importlib.import_module('layers.' + type)
    methodToCall = getattr(module_instance, 'build_layer_fn')
    return methodToCall(params)


