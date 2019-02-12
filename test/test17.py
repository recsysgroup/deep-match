import tensorflow as tf
import numpy as np

t = tf.placeholder(tf.int32, None)

a = tf.ones_like(t)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    print t.get_shape()
    print session.run(a, feed_dict={t: [0, 1, 0]})
