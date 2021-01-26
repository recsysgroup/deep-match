import tensorflow as tf


def optimizer_fn(config, train_max_step):
    learning_rate = tf.constant(value=config["learning_rate"], shape=[], dtype=tf.float32)
    opt = tf.train.AdagradOptimizer(learning_rate)
    return opt
