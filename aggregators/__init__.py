import importlib


def build_aggregator_fn(type, params):
    module_instance = importlib.import_module('aggregators.' + type)
    methodToCall = getattr(module_instance, 'build_aggregator_fn')
    return methodToCall(params)
