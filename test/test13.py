import tensorflow as tf
import numpy as np

c = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)

d = tf.constant([[0.2],[0.3]])

c1 = tf.matmul(c, d)
print c.get_shape()
print d.get_shape()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # print session.run(c)

    # print session.run(c1)
