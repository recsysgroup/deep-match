import tensorflow as tf
import numpy as np

v = tf.get_variable('xaaa', initializer=tf.zeros([100, 5]))



with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    print session.run(v)

