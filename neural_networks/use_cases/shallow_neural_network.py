import matplotlib.pyplot as plt
import numpy as np

from neural_networks.core.shallow_neural_network import ShallowNeuralNetwork

def function(x):
    return 5 * np.sin(x)


def learn_cuadratic_function():
    shallow_network = ShallowNeuralNetwork(2, 100)

    # square 10x10 between (-5, 5) in x and (-5, 5) in y
    x = 10 * np.random.rand(2, 1000) - 5

    z = np.ones((1, x.shape[-1]))
    z[:, x[1] < function(x[0])] = 0

    shallow_network.train(x, z, iterations=10000, learning_rate=0.1)
    is_over = shallow_network.activation(x)

    color = np.where(is_over < 0.5, "b", "r").squeeze()

    t = np.linspace(-5, 5, 100)
    plt.scatter(x[0], x[1], c=color, s=5, linewidth=0)
    plt.plot(t, function(t), "--")
    plt.ylim(np.min(x[1]), np.max(x[1]))
    plt.show()


    ###############
    x = 10 * np.random.rand(2, 1000) - 5
    is_over = shallow_network.activation(x)

    color = np.where(is_over < 0.5, "b", "r").squeeze()

    t = np.linspace(-5, 5, 100)
    plt.scatter(x[0], x[1], c=color, s=5, linewidth=0)
    plt.plot(t, function(t), "--")
    plt.ylim(np.min(x[1]), np.max(x[1]))
    plt.show()