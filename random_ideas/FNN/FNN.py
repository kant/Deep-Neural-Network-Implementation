'''FNN: Fully Connect Neural Network in which each layer connected
 to all layers before, instead of only the layer immediately before.

In many ways FNN is a generalized version of ResNet, and
 many other more complicated NN architecture like GoogLeNet

Intuition of FNN: sometimes, we need not only the last layer to predict things, and
certain less abstract information from earlier layer is useful as well.

As imagine, FNN has a lot of parameter but outperform traditional MLP easy, but
its highly subject to overfitting. Below a quick implementation on MNIST with Acc around 98.8%

Next Step: since there are just too many parameters,
we need to use some way to simplify FNN to be feasible to train, for example add a Conv Layer
 everytime access data from layer before, or use some systematic ways to break some connection in FNN.
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Layer 1: 784, 512 fully connected layer. Use input x
w0 = weight_variable([784, 512])
b0 = bias_variable([512])
a0 = tf.nn.relu(tf.matmul(x, w0) + b0)
keep_prob = tf.placeholder(tf.float32)
a0 = tf.nn.dropout(a0, keep_prob)

# Layer 2: 512, 512 fully connected layer.
# Use input x, a0 from layer 1
wx1 = weight_variable([784, 512])
w1 = weight_variable([512, 512])
b1 = bias_variable([512])
a1 = tf.nn.relu(tf.matmul(a0, w1) + b1 + tf.matmul(x, wx1))
a1 = tf.nn.dropout(a1, keep_prob)

# Layer 3: 512, 256 fully connected layer.
# Use input x, a0 from layer 1, a1 from layer 2
wx2 = weight_variable([784, 256])
wa0 = weight_variable([512, 256])
w2 = weight_variable([512, 256])
b2 = bias_variable([256])
a2 = tf.nn.relu(tf.matmul(a1, w2) + b2 + tf.matmul(x, wx2) + tf.matmul(a0, wa0))
a2 = tf.nn.dropout(a2, keep_prob)

# Layer 4: 256, 10 fully connected softmax output layer.
# Use input x, a0 from layer 1, a1 from layer 2, a3 from layer 3
wx3 = weight_variable([784, 10])
wa01 = weight_variable([512, 10])
wa10 = weight_variable([512, 10])
w3 = weight_variable([256, 10])
b3 = bias_variable([10])
y_nn = tf.nn.softmax(tf.matmul(a2, w3) + b3 + tf.matmul(x, wx3) + tf.matmul(a0, wa01) + tf.matmul(a1, wa10))


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_nn))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(128)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            print("Train Acc: ", train_accuracy, " Test Acc: ", test_acc)

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1}))
