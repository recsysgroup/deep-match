import tensorflow as tf
import numpy as np

labels = tf.constant([0, 0, 0, 1, 1, 1], dtype=tf.int32)
predictions = tf.constant([0.1, 0.3, 0.2, 0.6, 0.5, 0.2], dtype=tf.float32)

auc = tf.metrics.auc(labels, predictions)
mean = tf.metrics.mean(predictions)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())  # try commenting this line and you'll get the error

    l1 = session.run(auc)
    print l1

    l2 = session.run(auc)
    print l1,l2

    l3 = session.run(mean)
    l3 = session.run(mean)
    print l3
