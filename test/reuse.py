import tensorflow as tf
import numpy as np

t = tf.placeholder(tf.int32, None)

tensor = tf.Variable([[0, 1], [2, 3]], dtype=tf.float32)


def func(tensor):
    with tf.variable_scope('aaa', reuse=tf.AUTO_REUSE):
        tensor = tf.layers.dense(tensor, 2, name='ttt')
        print tensor
    return tensor


a = func(tensor)
b = func(tensor)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    print session.run([a, b])
