import tensorflow as tf
import numpy as np

c = tf.constant([1, 2])

cc = tf.constant([[0, 1, 2], [1, 2, 3]])

emb1 = tf.constant([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],dtype=tf.float32)

c = tf.expand_dims(c, axis=1)

c = tf.concat([c - tf.constant(1), c, c + tf.constant(1)], axis=1)

lp1 = tf.nn.embedding_lookup(emb1, c)

lp2 = tf.transpose(lp1, perm=[0, 2, 1])

lp3 = tf.reduce_mean(lp2, axis=-1)
# lp2 =

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    print session.run(c)

    print session.run([lp1])

    print session.run([lp2])


    print session.run([lp3])
