import numpy as np

from neural_networks.activations import relu


class Perceptron:
    def __init__(self, input_size: int, activation: callable = relu):
        self.activation = activation
        self.weights, self.bias = self.initialize_parameters(input_size)

    def initialize_parameters(self, input_size: int):
        weights = np.zeros((1, input_size))
        bias = 0

        return weights, bias

    def activation(self, x):
        return self.activation(np.dot(self.weights, x) + self.bias)

    def train(
        self, x, y, iterations: int = 1000, learning_rate: float = 0.1
    ):
        for i in range(iterations):
            pass
