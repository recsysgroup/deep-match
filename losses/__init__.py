import importlib
import constant as C

def build_loss_fn(config):
    _loss_params = config.get(C.CONFIG_LOSS)
    _loss_name = _loss_params.get(C.CONFIG_LOSS_NAME)
    module_instance = importlib.import_module('losses.' + _loss_name)
    methodToCall = getattr(module_instance, 'build_loss_fn')
    return methodToCall(config)


