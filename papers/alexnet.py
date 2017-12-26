import tensorflow as tf
import numpy as np

data = np.random.randn(200, 150528)
labels = np.random.randint(1000, size=200)
x = tf.placeholder(tf.float32, shape=[None, 150528])
y = tf.placeholder(tf.float32, shape=[None, 1000])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def inference(x):
    with tf.name_scope('hidden1'):
        weight = weight_variable([11, 11, 3, 96])
        bias = bias_variable([96])
        x_image = tf.reshape(x, [-1, 64, 64, 3])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, weight, strides=[1, 4, 4, 1], padding='SAME') + bias)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('hidden2'):
        weight = weight_variable([5, 5, 96, 256])
        bias = bias_variable([256])
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, weight, strides=[1, 1, 1, 1], padding='SAME') + bias)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('hidden3'):
        weight = weight_variable([3, 3, 256, 384])
        bias = bias_variable([384])
        h_con3 = tf.nn.relu(tf.nn.conv2d(h_pool2, weight, strides=[1, 1, 1, 1], padding='SAME') + bias)

    with tf.name_scope('hidden4'):
        weight = weight_variable([3, 3, 384, 384])
        bias = bias_variable([384])
        h_con4 = tf.nn.relu(tf.nn.conv2d(h_con3, weight, strides=[1, 1, 1, 1], padding='SAME') + bias)

    with tf.name_scope('hidden5'):
        weight = weight_variable([3, 3, 384, 256])
        bias = bias_variable([256])
        h_con5 = tf.nn.relu(tf.nn.conv2d(h_con4, weight, strides=[1, 1, 1, 1], padding='SAME') + bias)
        h_pool5 = tf.nn.max_pool(h_con5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('fully1'):
        weight = weight_variable([9216, 4096])
        bias = bias_variable([4096])
        res = tf.reshape(h_pool5, [-1, 9216])
        fully1 = tf.nn.relu((tf.matmul(res, weight) + bias))

    with tf.name_scope('fully2'):
        weight = weight_variable([4096, 4096])
        bias = bias_variable([4096])
        fully2 = tf.nn.relu((tf.matmul(fully1, weight) + bias))

    with tf.name_scope('out'):
        weight = weight_variable([4096, 1000])
        bias = bias_variable([1000])
        y_nn = tf.nn.softmax((tf.matmul(fully2, weight) + bias))

    return y_nn


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    y = inference(x)
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        train_step.run(feed_dict={x: x, y: inference(x)})
