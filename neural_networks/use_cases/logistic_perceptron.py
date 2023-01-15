import matplotlib.pyplot as plt
import numpy as np

from neural_networks.core.perceptron import LogisticPerceptron


def learn_logical_and():
    perceptron = LogisticPerceptron(input_size=2)

    perceptron.train(
        np.array([[1, 1], [1, 0], [0, 1], [0, 0]]).T,
        np.array([[1, 0, 0, 0]])
    )

    print(perceptron.activation(np.array([1, 1]).T))
    print(perceptron.activation(np.array([1, 0]).T))
    print(perceptron.activation(np.array([0, 1]).T))
    print(perceptron.activation(np.array([0, 0]).T))


def learn_logical_or():
    perceptron = LogisticPerceptron(input_size=2)

    perceptron.train(
        np.array([[1, 1], [1, 0], [0, 1], [0, 0]]).T,
        np.array([[1, 1, 1, 0]])
    )

    print(perceptron.activation(np.array([1, 1]).T))
    print(perceptron.activation(np.array([1, 0]).T))
    print(perceptron.activation(np.array([0, 1]).T))
    print(perceptron.activation(np.array([0, 0]).T))


def learn_logical_xor():
    perceptron = LogisticPerceptron(input_size=2)

    perceptron.train(
        np.array([[1, 1], [1, 0], [0, 1], [0, 0]]).T,
        np.array([[0, 1, 1, 0]])
    )

    print(perceptron.activation(np.array([1, 1]).T))
    print(perceptron.activation(np.array([1, 0]).T))
    print(perceptron.activation(np.array([0, 1]).T))
    print(perceptron.activation(np.array([0, 0]).T))
