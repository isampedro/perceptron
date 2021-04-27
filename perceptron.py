import numpy as np
import random
import json
import calculateError as cError

def simplePerceptron(entry, expectedExit, N, K, eta, calculateError, g, MAX_ROUNDS, beta, errorMinStart, deltaWFunc, gder, exercise, deltaError, entryTesting, expectedExitTesting, trainingQ):
    for elem in entry:
        elem.insert(0, 1)
    for elem in entryTesting:
        elem.insert(0, 1)
    entryArr = np.array(entry)
    entryTestingArr = np.array(entryTesting)
    expectedExitArr = np.array(expectedExit)
    expectedExitTestingArr = np.array(expectedExitTesting)
    w = np.zeros(N+1)
    error = 1
    error_min = errorMinStart
    errors = []
    errorsTesting = []
    accuracies = []
    accuraciesTesting = []
    i = 0
    while error > 0 and i < MAX_ROUNDS:
        i_k = i % (int(K*trainingQ)-1)
        h = np.dot(entryArr[i_k], w)
        deltaW = deltaWFunc(eta, g, gder, h, expectedExitArr[i_k][0], entryArr[i_k], beta)
        w = w + deltaW
        error = calculateError(entryArr, expectedExitArr, w, g, int(K*trainingQ), beta)
        errors.append({'error': error, 'iteration': i})
        error = calculateError(entryTestingArr, expectedExitTestingArr, w, g, int(K*(1-trainingQ)), beta)
        errorsTesting.append({'error': error, 'iteration': i})
        accuracy = cError.accuracy(entryArr, expectedExitArr, w, g, int(K*trainingQ), beta, deltaError)
        accuracies.append(accuracy)
        accuracy = cError.accuracy(entryTestingArr, expectedExitTestingArr, w, g, int(K*(1-trainingQ)), beta, deltaError)
        accuraciesTesting.append(accuracy)
        if error < error_min:
            error_min = error
            w_min = w
        i = i + 1
    return { 'errors': errors, 'errorsTesting': errorsTesting, 'w_min': w_min.tolist(), 'accuracyTesting': accuracies, 'accuracyTraining': accuraciesTesting }