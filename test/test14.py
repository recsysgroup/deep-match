import tensorflow as tf
import numpy as np

e1 = tf.constant([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=tf.float32)

e2 = tf.constant([[5, 6, 5, 6], [7, 8, 5, 6]], dtype=tf.float32)

e3 = tf.constant([[55, 66, 5, 6], [77, 88, 5, 6]], dtype=tf.float32)

# [B, F, E]
c = tf.stack([e1, e2, e3], axis=1)
print 'c' + str(c.get_shape())

s = tf.slice(tf.shape(c), [0], [2])

# [B, F]
r = tf.random_uniform(s)
r = 0.8 + r

# [B, F]
b = tf.cast(tf.floor(r), dtype=tf.float32)
print 'b:' + str(r.get_shape())

b = tf.tile(tf.expand_dims(b,-1), [1, 1, 4])

e = tf.multiply(c, b)


with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    l = session.run([c,b,e])


    for i in l:
        print '###'
        print i

