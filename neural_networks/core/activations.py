import numpy as np


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, scaler: float = 0.01):
    return np.maximum(scaler * x, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def backward_relu(x):
    return x > 0


def backward_leaky_relu(x, scaler: float = 0.01):
    dx = np.ones_like(x)
    dx[x < 0] = scaler
    return dx


def backward_sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig * (1 - sig)
