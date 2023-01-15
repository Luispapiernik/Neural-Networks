import numpy as np

from neural_networks.core.activations import relu
from neural_networks.core.costs import logistic_cost


class Perceptron:
    def __init__(self, input_size: int, activation: callable = relu):
        self.activation_function = activation
        self.weights, self.bias = self.initialize_parameters(input_size)

    def initialize_parameters(self, input_size: int):
        weights = np.zeros((1, input_size))
        bias = 0

        return weights, bias

    def activation(self, x):
        return self.activation_function(np.dot(self.weights, x) + self.bias)

    def train(
        self, x, y, iterations: int = 100000, learning_rate: float = 0.001
    ):
        m = y.shape[-1]
        for i in range(iterations):
            a = self.activation(x)
            cost = logistic_cost(a, y)

            dz = a - y
            db = np.sum(dz) / m
            dw = np.sum(x * dz, axis=1, keepdims=True).T / m

            self.bias = self.bias - learning_rate * db
            self.weights = self.weights - learning_rate * dw
