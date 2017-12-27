"""
Implement of ResNet from "Deep Residual Learning for Image Recognition" on cifar10
20 Layers ResNet as described in section 4.2
Acc reaches 90% after 64k steps, without any data augmentation
"""


import tensorflow as tf
import numpy as np
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
    k = float(shape[0])
    c = float(shape[-1])
    std = np.sqrt(2 / (k ** 2 * c))

    init = tf.truncated_normal(shape, stddev=std)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1, 32, 32, 3])


def conv3_3(inputs, num_filters):
    shape = inputs.shape
    weight = weight_variable([3, 3, int(shape[-1]), num_filters])
    bias = bias_variable([num_filters])
    conv = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding="SAME")

    batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=True)
    act = tf.nn.relu(batch_norm + bias)

    return act


layer11 = conv3_3(x_image, 16)
layer12 = conv3_3(layer11 + x_image, 16)
layer13 = conv3_3(layer12 + layer11, 16)
layer14 = conv3_3(layer13 + layer12, 16)
layer15 = conv3_3(layer14 + layer13, 16)
layer16 = conv3_3(layer15 + layer14, 16)
layer17 = conv3_3(layer16 + layer15, 16)
layer1_pool = tf.nn.max_pool(layer17, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

layer21 = conv3_3(layer1_pool, 32)
layer22 = conv3_3(layer21 + layer1_pool, 32)
layer23 = conv3_3(layer22 + layer21, 32)
layer24 = conv3_3(layer23 + layer22, 32)
layer25 = conv3_3(layer24 + layer23, 32)
layer26 = conv3_3(layer25 + layer24, 32)

layer2_pool = tf.nn.max_pool(layer26, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

layer31 = conv3_3(layer2_pool, 64)
layer32 = conv3_3(layer31 + layer2_pool, 64)
layer33 = conv3_3(layer32 + layer31, 64)
layer34 = conv3_3(layer33 + layer32, 64)
layer35 = conv3_3(layer34 + layer33, 64)
layer36 = conv3_3(layer35 + layer34, 64)

flatten = tf.reshape(layer36, [-1, 4096])
weight5 = weight_variable([4096, 10])
bias5 = bias_variable([10])

out = tf.matmul(flatten, weight5) + bias5

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

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1

global_step = tf.Variable(0, trainable=False)
boundaries = [32000, 48000]
values = [0.001, 0.0005, 0.0002]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_acc = []
dev_acc = []


with tf.device('/gpu:0'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(64000):
            batch = get_batch(x_train, y_train, 128)
            if i % 200 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
                test_acc = accuracy.eval(feed_dict={x: x_test, y: y_test})
                train_acc.append(train_accuracy)
                dev_acc.append(test_acc)
                print("Iteration: ", i, "Train: ", train_accuracy, " Test: ", test_acc)
            train_step.run(feed_dict={x: batch[0], y: batch[1]})
