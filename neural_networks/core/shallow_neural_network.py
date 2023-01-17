import numpy as np

from neural_networks.core.activations import sigmoid, relu


class ShallowNeuralNetwork:
    def __init__(self, input_size: int, hiden_layer_size: int) -> None:
        self.input_size = input_size
        self.hiden_layer_size = hiden_layer_size
        self.reset_parameters()

    def initialize_parameters(self, input_size: int, hiden_layer_size: int):
        w1 = np.random.randn(hiden_layer_size, input_size) * 0.01
        b1 = np.zeros((hiden_layer_size, 1))
        w2 = np.random.randn(1, hiden_layer_size) * 0.01
        b2 = np.zeros((1, 1))

        parameters = {
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2
        }
        return parameters

    def reset_parameters(self) -> None:
        self.parameters = self.initialize_parameters(
            self.input_size, self.hiden_layer_size
        )

    def activation(self, x, with_cache: bool = False):
        w1 = self.parameters["w1"]
        b1 = self.parameters["b1"]
        w2 = self.parameters["w2"]
        b2 = self.parameters["b2"]

        z1 = np.dot(w1, x) + b1
        a1 = np.tanh(z1)

        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)
        if with_cache:
            return a2, {"z1": z1, "a1": a1, "z2": z2, "a2": a2}

        return a2

    def backward(self, x, y, cache):
        m = y.shape[-1]

        a1 = cache["a1"]
        a2 = cache["a2"]
        w2 = self.parameters["w2"]

        dz2 = a2 - y
        dw2 = np.dot(dz2, a1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = np.dot(w2.T, dz2) * (1 - np.power(a1, 2))
        dw1 = np.dot(dz1, x.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        return {
            "dw1": dw1,
            "db1": db1,
            "dw2": dw2,
            "db2": db2
        }

    def train(
        self, x, y, iterations: int = 1000, learning_rate: float = 0.05
    ):
        for _ in range(iterations):
            _, cache = self.activation(x, with_cache=True)

            dp = self.backward(x, y, cache)

            self.parameters["w1"] = self.parameters["w1"] - learning_rate * dp["dw1"]
            self.parameters["b1"] = self.parameters["b1"] - learning_rate * dp["db1"]
            self.parameters["w2"] = self.parameters["w2"] - learning_rate * dp["dw2"]
            self.parameters["b2"] = self.parameters["b2"] - learning_rate * dp["db2"]
