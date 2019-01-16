import tensorflow as tf
import numpy as np


def _decode_record(line):
    return {'d1': line, 'd2': line}


d1 = tf.data.TextLineDataset('/tmp/a.txt')
d1 = d1.repeat()
d1 = d1.shuffle(100)
d1 = d1.batch(2)

d2 = tf.data.TextLineDataset('/tmp/b.txt')
d2 = d2.repeat()
d2 = d2.shuffle(100)
d2 = d2.map(_decode_record)
d2 = d2.batch(3)

d = tf.data.Dataset.zip((d1, d2))

iterator = d.make_one_shot_iterator()
one_element = iterator.get_next()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    session.run(tf.tables_initializer())

    for i in range(5):
        print session.run(one_element)
