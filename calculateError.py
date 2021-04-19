import numpy as np

def squareError(entryArr, expectedExitArr, w, g, N, beta):
    squareErrorVal = 0
    for i in range(N):
        h = np.dot(entryArr[i], w)
        delta = g(h, beta) - expectedExitArr[i][0]
        squareErrorVal += delta*delta
    squareErrorVal = squareErrorVal/N
    return squareErrorVal

def accuracy( entryArr, expectedExitArr, w, g, N, beta, deltaError):
    corrects = 0
    for i in range(N):
        h = np.dot(entryArr[i], w)
        if g(h, beta) >= expectedExitArr[i][0] - deltaError and g(h, beta) <= expectedExitArr[i][0] + deltaError:
            corrects = corrects + 1
    return corrects/N
