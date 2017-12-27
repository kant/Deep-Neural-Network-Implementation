'''
Basic CNN implementation with TensorFlow, with idea from Keras
https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py

Reach 75% on CIFAR10 after 25 Epochs
'''


import tensorflow as tf
import keras
from keras.datasets import cifar10

num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train /= 255
x_test /= 255


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])

# Layer1
x_image = tf.reshape(x, [-1, 32, 32, 3])
weight1 = weight_variable([3, 3, 3, 32])
bias1 = bias_variable([32])
conv1 = tf.nn.relu(tf.nn.conv2d(x_image, weight1, strides=[1, 1, 1, 1], padding="SAME") + bias1)

weight2 = weight_variable([3, 3, 32, 32])
bias2 = bias_variable([32])
conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weight2, strides=[1, 1, 1, 1], padding="VALID") + bias2)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
keep_prob = tf.placeholder(tf.float32)
pool2 = tf.nn.dropout(pool2, keep_prob)

weight3 = weight_variable([3, 3, 32, 64])
bias3 = bias_variable([64])
conv3 = tf.nn.relu(tf.nn.conv2d(pool2, weight3, strides=[1, 1, 1, 1], padding="SAME") + bias3)

weight4 = weight_variable([3, 3, 64, 64])
bias4 = bias_variable([64])
conv4 = tf.nn.relu(tf.nn.conv2d(conv3, weight4, strides=[1, 1, 1, 1], padding="VALID") + bias4)
pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
pool4 = tf.nn.dropout(pool4, keep_prob)

flatten = tf.reshape(pool4, [-1, 2304])
weight5 = weight_variable([2304, 512])
bias5 = bias_variable([512])
z5 = tf.nn.relu(tf.matmul(flatten, weight5) + bias5)

weight6 = weight_variable([512, 10])
bias6 = bias_variable([10])
out = tf.matmul(z5, weight6) + bias6

global counter
counter = 0


def get_batch(x_in, y_in, batch_size):
    global counter

    upper = counter + batch_size
    if upper >= len(x_in):
        res = x_in[counter:], y_in[counter:]
        counter = 0
    else:
        res = x_in[counter:upper], y_in[counter:upper]
        counter = upper
    return res


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200000):
        batch = get_batch(x_train, y_train, 32)
        if i % 2000 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1})
            test_acc = accuracy.eval(feed_dict={x: x_test, y: y_test, keep_prob: 1.0})
            print("Iteration: ", i, "Train: ", train_accuracy, " Test: ", test_acc)
        train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8})
