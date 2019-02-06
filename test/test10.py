import tensorflow as tf
import numpy as np

c = tf.constant(['1', '2', '3'])
c1 = tf.string_to_number(c, tf.float32)
c2 = tf.to_float(c)

a = {'price': tf.constant([9.0])}
price = tf.feature_column.numeric_column('price')
bucketized_price = tf.feature_column.bucketized_column(price, [0, 1, 2, 3, 4, 5])
price_bucket_tensor = tf.feature_column.input_layer(a, [bucketized_price])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    print session.run(price_bucket_tensor)
