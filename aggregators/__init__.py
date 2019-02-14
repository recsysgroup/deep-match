import importlib


def build_aggregator_fn(type, params):
    if type is None:
        type = 'sum'
    module_instance = importlib.import_module('aggregators.' + type)
    methodToCall = getattr(module_instance, 'build_aggregator_fn')
    return methodToCall(params)
