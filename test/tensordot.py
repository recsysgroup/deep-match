import tensorflow as tf
import numpy as np

a = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], shape=[2, 2, 3], dtype=tf.float32)

b = tf.constant([1, 1, 1, 1, 1, 1], shape=[2, 3], dtype=tf.float32)

c = tf.tensordot(a, b, axes=[[1], [0]])

d =tf.squeeze(tf.matmul(a, tf.expand_dims(b, -1)))


with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    print session.run([c, d])
