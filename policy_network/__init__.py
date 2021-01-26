import importlib


def build_policy_fn(type, params, name):
    module_instance = importlib.import_module('policy_network.' + type)
    methodToCall = getattr(module_instance, 'build_policy_fn')
    return methodToCall(params, name)


