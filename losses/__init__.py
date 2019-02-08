import importlib


def build_loss_fn(type, params):
    module_instance = importlib.import_module('losses.' + type)
    methodToCall = getattr(module_instance, 'build_loss_fn')
    return methodToCall(params)


