import numpy as np

from neural_networks.core.activations import sigmoid

class LogisticPerceptron:
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.reset_parameters()

    def initialize_parameters(self, input_size: int) -> None:
        weights = np.random.randn(input_size, 1)
        bias = 0

        return weights, bias
    
    def reset_parameters(self) -> None:
        self.weights, self.bias = self.initialize_parameters(self.input_size)

    def activation(self, x):
        return sigmoid(np.dot(self.weights.T, x) + self.bias)

    def backward(self, x, a, y):
        dp = {}
        m = y.shape[-1]

        dz = a - y
        db = np.sum(dz) / m
        dw = np.dot(x, dz.T) / m

        dp["dw"] = dw
        dp["db"] = db

        return dp

    def train(
        self, x, y, iterations: int = 1000, learning_rate: float = 0.05
    ):
        for _ in range(iterations):
            a = self.activation(x)

            dp = self.backward(x, a, y)
            db = dp["db"]
            dw = dp["dw"]

            self.bias = self.bias - learning_rate * db
            self.weights = self.weights - learning_rate * dw
