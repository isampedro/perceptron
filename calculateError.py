import numpy as np

def squareError(entryArr, expectedExitArr, w, g, N):
    squareErrorVal = 0
    for i in range(N):
        h = np.dot(entryArr[i], w)
        squareErrorVal += np.power(g(h)-expectedExitArr[i], 2)
    squareErrorVal = squareErrorVal/N
    return squareErrorVal
