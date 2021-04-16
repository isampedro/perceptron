import numpy as np

def cuadraticError(h, yArr, N):
    return np.sum(np.power(h-yArr,2))/N
