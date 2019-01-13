import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

x = tf.keras.layers.Dense(128, activation='relu')(img)
x = tf.keras.layers.Dense(128, activation='relu')(x)
print x
prediction = tf.keras.layers.Dense(10, activation='softmax')(x)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=labels))

train_optim = tf.train.AdamOptimizer().minimize(loss)

mnist_data = input_data.read_data_sets('MNIST_data/', one_hot=True)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
sess.run(init)

for _ in range(1000):
    batch_x, batch_y = mnist_data.train.next_batch(50)
sess.run(train_optim, feed_dict={img: batch_x, labels: batch_y})

acc_pred = tf.keras.metrics.categorical_accuracy(labels, prediction)
pred = sess.run(acc_pred, feed_dict={labels: mnist_data.test.labels, img: mnist_data.test.images})

print('accuracy: %.3f' % (sum(pred) / len(mnist_data.test.labels)))