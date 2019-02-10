import tensorflow as tf
import numpy as np

e1 = tf.constant([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=tf.float32)

e2 = tf.constant([[5, 6, 5, 6], [7, 8, 5, 6]], dtype=tf.float32)

e3 = tf.constant([[55, 66, 5, 6], [77, 88, 5, 6]], dtype=tf.float32)

ee = {'e1': e1, 'e2': e2, 'e3': e3}

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    l = session.run(ee)

    print l
