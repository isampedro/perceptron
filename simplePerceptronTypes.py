import numpy as np

def scalar(x, beta):
    if x >= 0:
        return 1
    return -1

def nonLinear(x, beta):
    return np.tanh(beta*x)