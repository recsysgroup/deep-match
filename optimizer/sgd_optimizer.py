import tensorflow as tf


def optimizer_fn(config, train_max_step):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value=config["learning_rate"], shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        train_max_step,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    tf.summary.scalar('learning_rate', learning_rate)

    opt = tf.train.GradientDescentOptimizer(learning_rate)
    return opt
