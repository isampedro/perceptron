import numpy as np

def squareError(entryArr, expectedExitArr, w, g, N):
    squareErrorVal = 0
    for i in range(N):
        h = np.dot(entryArr[i], w)
        print('entryArr[i]', entryArr[i])
        print('w: ', w)
        print('h: ', h)
        print('g(h): ', g(h))
        print('expectedExitArr[i]: ',expectedExitArr[i])
        delta = g(h) - expectedExitArr[i]
        print('delta: ', delta)
        squareErrorVal += np.dot(delta, delta)
        print('square val: ', squareErrorVal)
    squareErrorVal = squareErrorVal/N
    return squareErrorVal
