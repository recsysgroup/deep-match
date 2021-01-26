import importlib
import constant as C
import tensorflow as tf


def build_optimizer_fn(config):
    optimizer_name = config['optimizer']

    if config.get(C.BIZ_NAME) is not None:
        try:
            biz_name = config.get(C.BIZ_NAME)
            path = '.'.join(['biz', biz_name, optimizer_name])
            module_instance = importlib.import_module(path)
            methodToCall = getattr(module_instance, 'optimizer_fn')
            tf.logging.info('register sampler from {}'.format(path))
            return methodToCall
        except ImportError:
            pass

    module_instance = importlib.import_module('optimizer.' + optimizer_name)
    methodToCall = getattr(module_instance, 'optimizer_fn')
    return methodToCall
