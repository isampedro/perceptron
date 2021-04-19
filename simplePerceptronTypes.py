import numpy as np

def scalar(x, beta):
    if x >= 0:
        return 1
    return -1

def linear(x, beta):
    return x

def nonLinear(x, beta):
    return np.tanh(beta*x)

def nonLinearDer(x, beta):
    return beta/np.cosh(beta*x)**2