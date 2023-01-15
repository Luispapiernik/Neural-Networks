import numpy as np


def logistic_loss(a, y):
    return y * np.log(a) + (1 - y) * np.log(1 - a)


def logistic_cost(a, y):
    return - np.sum(logistic_loss(a, y)) / a.shape[-1]
