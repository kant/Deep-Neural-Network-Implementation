import numpy as np


class Logistic(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.weights = np.zeros((x.shape[1], 1))

    def sigmoid(self, score):
        return 1 / (1 + np.exp(-score))

    def loss_func(self):
        scores = np.dot(self.x, self.weights)
        ll = np.sum(self.y * scores - np.log(1 + np.exp(scores)))
        return ll

    def gradient_decent(self, lr, iter_num):
        for i in range(iter_num):
            scores = np.dot(self.x, self.weights)
            scores_prob = self.sigmoid(scores)
            error = self.y - scores_prob
            gradient = lr * np.dot(self.x.T, error)
            self.weights += gradient
            print(self.loss_func())

    def stochastic_gradient_decent(self, lr):
        for cur_x, cur_y in zip(self.x, self.y):
            scores = np.dot(cur_x, self.weights)
            scores_prob = self.sigmoid(scores)
            error = cur_y - scores_prob
            gradient = (lr * cur_x * error).reshape(10, 1)
            self.weights += gradient

    def train(self, lr, iter_num):
        self.gradient_decent(lr, iter_num)

    def predict(self, x_test):
        scores = (np.dot(x_test, self.weights))
        prob = self.sigmoid(scores)
        prob[prob >= 0.5] = 1
        prob[prob <= 0.5] = 0
        return prob
