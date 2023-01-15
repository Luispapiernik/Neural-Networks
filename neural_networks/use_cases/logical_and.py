import numpy as np

from neural_networks.core.perceptron import Perceptron
from neural_networks.core.activations import sigmoid


def learn_logical_and():
    perceptron = Perceptron(input_size=2, activation=sigmoid)

    perceptron.train(
        np.array([[1, 1], [1, 0], [0, 1], [0, 0]]).T,
        np.array([[1, 0, 0, 0]])
    )

    print(perceptron.activation(np.array([1, 1]).T))
    print(perceptron.activation(np.array([1, 0]).T))
    print(perceptron.activation(np.array([0, 1]).T))
    print(perceptron.activation(np.array([0, 0]).T))

    perceptron = Perceptron(input_size=2, activation=sigmoid)

    perceptron.train(
        np.array([[1, 1], [1, 0], [0, 1], [0, 0]]).T,
        np.array([[1, 1, 1, 0]])
    )

    print(perceptron.activation(np.array([1, 1]).T))
    print(perceptron.activation(np.array([1, 0]).T))
    print(perceptron.activation(np.array([0, 1]).T))
    print(perceptron.activation(np.array([0, 0]).T))
