import tensorflow as tf
import numpy as np

c = tf.constant(['1', '2', '3'])
c1 = tf.string_to_number(c, tf.float32)
c2 = tf.to_float(c)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    print session.run([c1, c2])
