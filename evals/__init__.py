import importlib


def build_eval_fn(type, params):
    module_instance = importlib.import_module('evals.' + type)
    methodToCall = getattr(module_instance, 'build_eval_fn')
    return methodToCall(params)

