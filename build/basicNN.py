import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    score = sigmoid(x)
    return score * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    s = np.maximum(x, 0)
    return s


def relu_deriv(x):
    x[x < 0] = 0
    x[x >= 0] = 1
    return x


func_dic = {"sigmoid": sigmoid, "tanh": tanh, "relu": relu}
deriv_dic = {"sigmoid": sigmoid_deriv, "tanh": tanh_deriv, "relu": relu_deriv}


class Layer(object):
    def __init__(self, num_node, num_node_prev, activation, derivative, keepprob=1):
        self.num_node = num_node
        self.activation = activation
        self.derivative = derivative
        self.output_layer = False
        self.a = 0
        self.z = 0
        self.weight = np.random.randn(num_node, num_node_prev) * np.sqrt(2 / num_node_prev)
        self.b = np.zeros((num_node, 1))
        self.dw = 0
        self.db = 0
        self.keepprob = keepprob

    def forward(self, x):
        self.z = np.dot(self.weight, x) + self.b
        self.a = self.activation(self.z)
        dropout = np.random.rand(self.a.shape[0], self.a.shape[1]) < self.keepprob
        dropped_a = self.a * dropout
        dropped_a /= self.keepprob
        return dropped_a

    def backward(self, da, a_prev, lr):
        if self.output_layer:
            dz = da
        else:
            dz = da * self.derivative(self.z)
        dw = (1 / da.shape[1]) * np.dot(dz, a_prev.T)
        db = (1 / da.shape[1]) * np.sum(dz, axis=1, keepdims=True)
        self.weight -= lr * dw
        self.dw = dw
        self.b -= lr * db
        self.db = db
        da_prev = np.dot(self.weight.T, dz)

        return da_prev


class NeuralNetwork(object):
    def __init__(self, x, y, x_dev, y_dev, layer_node_num, layer_activations, dropout=1):
        self.x = x
        self.y = y
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.dropout = dropout
        self.layers = self.build_layers(layer_node_num, layer_activations)

    def loss_func(self, x, y):
        m = x.shape[1]
        scores = self.forward_prop(x).T
        ll = np.sum(y * scores - scores)
        return (1 / m) * ll * 100

    def build_layers(self, layer_node_num, layer_activations):
        whole_layers = []
        prev = self.x.shape[0]
        for num, func in zip(layer_node_num, layer_activations):
            cur_layer = Layer(num, prev, func_dic[func], deriv_dic[func], self.dropout)
            whole_layers.append(cur_layer)
            prev = num
        whole_layers[-1].output_layer = True
        return whole_layers

    def forward_prop(self, x):
        prev_a = x
        for layer in self.layers:
            prev_a = layer.forward(prev_a)
        return prev_a

    def back_prop(self, lr):
        yhat = self.forward_prop(self.x)
        da = yhat - self.y.T
        i = len(self.layers) - 1
        while i >= 0:
            cur_layer = self.layers[i]
            if i == 0:
                a_prev = self.x
            else:
                a_prev = self.layers[i - 1].a
            da = cur_layer.backward(da, a_prev, lr)
            i -= 1

    def train(self, lr, iter_num):
        for i in range(iter_num):
            self.back_prop(lr)
            if i % 1000 == 0:
                print(self.loss_func(self.x, self.y), self.loss_func(self.x_dev, self.y_dev))

    def predict(self, x_test):
        for layer in self.layers:
            layer.keepprob = 1
        prob = self.forward_prop(x_test)
        prob[prob >= 0.5] = 1
        prob[prob <= 0.5] = 0
        return prob.T
