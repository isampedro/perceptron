import numpy as np
import random


def simplePerceptron(entry, expectedExit, N, K, eta, calculateError, g, MAX_ROUNDS):
    entryArr = np.array(entry)
    expectedExitArr = np.array(expectedExit)
    w = np.zeros(N+1)
    error = 1
    error_min = K*2
    i = 0
    while error > 0 and i < MAX_ROUNDS:
        i_k = random.randint(0, K-1)
        h = np.dot(entryArr[i_k], w)
        print('entry array: ', entryArr[i_k])
        print('w: ', w)
        print('h: ', h)
        g_h = g(h)
        print('g_h: ',g_h)
        deltaW = eta*(expectedExitArr[i_k] - g_h)*entryArr[i_k]
        w = w + deltaW
        error = calculateError(entryArr, expectedExitArr, w, g, K)
        print('error: ', error)
        if error < error_min:
            error_min = error
            w_min = w
        i = i + 1
    return w_min
