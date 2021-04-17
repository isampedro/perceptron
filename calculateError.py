import numpy as np

def squareError(entryArr, expectedExitArr, w, g, N, beta):
    squareErrorVal = 0
    for i in range(N):
        h = np.dot(entryArr[i], w)
        delta = g(h, beta) - expectedExitArr[i][0]
        squareErrorVal += np.dot(delta, delta)
    squareErrorVal = squareErrorVal/N
    return squareErrorVal