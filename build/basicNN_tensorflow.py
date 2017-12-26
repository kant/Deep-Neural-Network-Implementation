'''TensorFlow Implementation of Basic Neural Network'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


y_nn = tf.placeholder(tf.float32, shape=[None, 10])

# Define some functions
relu = tf.nn.relu
softmax = tf.nn.softmax


class Layer(object):
    def __init__(self, shape, activation, keepprob):
        self.shape = shape
        self.activation = activation
        self.w = weight_variable(self.shape)
        self.b = bias_variable([self.shape[1]])
        self.keepprob = keepprob

    def forward_prop(self, x):
        z = tf.matmul(x, self.w)
        a = self.activation(z + self.b)
        a_drop = tf.nn.dropout(a, self.keepprob)
        return a_drop


class NeuralNetwork(object):
    def __init__(self, x_train, y_train, x_dev, y_dev, layer_sizes, act_func, keepprob=1):
        self.x_train = x_train
        self.y_train = y_train
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.layer_sizes = layer_sizes
        self.act_func = act_func
        self.keepprob = keepprob
        self.layers = self.build()
        self.data_counter = 0
        self.sess = tf.Session()

    def batch_data(self, batch_size):
        counter = self.data_counter
        upper = counter + batch_size
        if upper < len(self.x_train):
            res = (self.x_train[counter:upper], self.y_train[counter:upper])
            self.data_counter = upper
        else:
            res = (self.x_train[counter:], self.y_train[counter:])
            self.data_counter = 0
        return res

    def build(self):
        res_layers = []
        prev = self.x_train.shape[1]
        for size, act in zip(self.layer_sizes, self.act_func):
            cur_layer = Layer(shape=(prev, size), activation=act, keepprob=self.keepprob)
            res_layers.append(cur_layer)
            prev = size

        return res_layers

    def predict(self, x):
        res = x
        for layer in self.layers:
            res = layer.forward_prop(res)
        return res

    def init_loss(self, y_nn):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_nn))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.accuracy = accuracy

    def train(self, batch_size, iter_num, verbose=1):
        y_nn = self.predict(x)
        self.init_loss(y_nn)
        self.sess.run(tf.global_variables_initializer())
        for i in range(iter_num):
            batch = self.batch_data(batch_size)
            if i % 100 == 1 & verbose:
                train_accuracy = self.accuracy.eval(session=self.sess,
                                                    feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                test_acc = self.accuracy.eval(session=self.sess,
                                              feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
                print("Train Acc: ", train_accuracy, " Test Acc: ", test_acc)

            self.sess.run(self.train_step, {x: batch[0], y_: batch[1], keep_prob: 0.5})


model = NeuralNetwork(mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels, [512, 256, 10], [relu, relu, softmax])
model.train(128, 20000)