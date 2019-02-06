import tensorflow as tf
import numpy as np

c = tf.constant([1, 2, 3])

c1 = c - tf.constant(3)

c2 = tf.minimum(c, tf.constant(2))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    print session.run([c1, c2])
