import tensorflow as tf
import numpy as np

a = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)

b = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)

c_list = []
for i in range(2):
    start = i + 1
    tmp1 = tf.slice(b, [start, 0], [3 - start, 2])
    tmp2 = tf.slice(b, [0, 0], [start, 2])
    tmp = tf.concat([tmp1, tmp2], axis=0)
    c_list.append(tf.expand_dims(tmp, axis=1))

c = tf.concat(c_list, axis=1)

d = tf.multiply(tf.expand_dims(a, axis=1), c)

e = tf.reduce_sum(d, axis=-1)
print c.get_shape()
print e.get_shape()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    print session.run(e)

    # print session.run(d)
