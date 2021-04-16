import numpy as np


def simplePerceptron(x, y, N, K, eta, calculateError, g):
    xArr = np.array(x)
    yArr = np.array(y)
    w = np.zeros(N+1)
    error = 1
    error_min = N*2
    i = 0
    while error > 0 and i < K:
        h = np.dot(xArr[i], w)
        deltaW = eta*(yArr[i] - g(h))*xArr[i]
        w += deltaW
        error = calculateError(h, yArr, N)
        if error < error_min:
            error_min = error
            ans = w
        i += 1
    return ans
