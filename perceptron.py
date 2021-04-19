import numpy as np
import random
import json
import calculateError as cError

def simplePerceptron(entry, expectedExit, N, K, eta, calculateError, g, MAX_ROUNDS, beta, errorMinStart, deltaWFunc, gder, exercise, deltaError):
    for elem in entry:
        elem.insert(0, 1)
    entryArr = np.array(entry)
    expectedExitArr = np.array(expectedExit)
    w = np.zeros(N+1)
    error = 1
    error_min = errorMinStart
    errors = []
    accuracies = []
    i = 0
    while error > 0 and i < MAX_ROUNDS:
        i_k = i % K
        h = np.dot(entryArr[i_k], w)
        deltaW = deltaWFunc(eta, g, gder, h, expectedExitArr[i_k][0], entryArr[i_k], beta)
        w = w + deltaW
        error = calculateError(entryArr, expectedExitArr, w, g, K, beta)
        errors.append({'error': error, 'iteration': i})
        accuracy = cError.accuracy(entryArr, expectedExitArr, w, g, K, beta, deltaError)
        accuracies.append(accuracy)
        if error < error_min:
            error_min = error
            w_min = w
        i = i + 1
    return { 'errors': errors, 'w_min': w_min.tolist(), 'accuracies': accuracies }