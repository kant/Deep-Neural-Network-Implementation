import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar10

num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=tf.sqrt(2 / shape[0]))
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 32, 32, 3])
weight1 = weight_variable([3, 3, 3, 6])
bias1 = bias_variable([6])
conv1 = tf.nn.relu(tf.nn.conv2d(x_image, weight1, strides=[1, 1, 1, 1], padding="VALID") + bias1)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

weight2 = weight_variable([5, 5, 6, 16])
bias2 = bias_variable([16])
conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weight2, strides=[1, 1, 1, 1], padding="VALID") + bias2)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

flatten = tf.reshape(pool2, [-1, 400])
weight3 = weight_variable([400, 120])
bias3 = bias_variable([120])
z3 = tf.nn.relu(tf.matmul(flatten, weight3) + bias3)

weight4 = weight_variable([120, 10])
bias4 = bias_variable([10])
out = tf.nn.softmax(tf.matmul(z3, weight4) + bias4)

global counter
counter = 0


def get_batch(x, y, batch_size):
    global counter
    upper = counter + batch_size
    if upper > len(x):
        res = x[counter:], y[counter:]
        counter = 0
    else:
        res = x[counter:upper], y[counter:upper]
        counter = upper
    return res


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200000):
        batch = get_batch(x_train, y_train, 32)
        if i % 1000 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
            print('step %d, training accuracy %g' % (i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y: batch[1]})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: x_test, y: y_test}))
