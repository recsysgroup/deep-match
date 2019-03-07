import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=[None, 3])

bias = tf.constant([0.1, 0.2, 0.3], shape=[1, 3])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print x.get_shape()
    t = tf.shape(x)
    b_bias = tf.tile(bias, [t[0],1])
    print session.run(t, feed_dict={x: [[1, 2, 3], [5, 6, 7]]})
    print b_bias.get_shape()
    print session.run(b_bias,feed_dict={x: [[1, 2, 3], [5, 6, 7]]})
