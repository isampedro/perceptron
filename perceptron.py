import numpy as np
import random
import json

def simplePerceptronJSON(entry, expectedExit, N, K, eta, calculateErrorJSON, g, MAX_ROUNDS, beta, errorMinStart):
    entryArr = np.array(entry)
    expectedExitArr = np.array(expectedExit)
    w = np.zeros(N+1)
    error = 1
    error_min = errorMinStart
    errors = []
    i = 0
    while error > 0 and i < MAX_ROUNDS:
        i_k = random.randint(0, K-1)
        h = np.dot(entryArr[i_k], w)
        g_h = g(h, beta)
        deltaW = eta*(expectedExitArr[i_k] - g_h)*entryArr[i_k]
        w = w + deltaW
        error = calculateErrorJSON(entryArr, expectedExitArr, w, g, K, beta)
        errors.append({'error': error, 'iteration': i})
        if error < error_min:
            error_min = error
            w_min = w
        i = i + 1
    return { 'errors': errors, 'w_min': w_min.tolist() }

def simplePerceptronPandas(entry, expectedExit, N, K, eta, calculateErrorPandas, g, MAX_ROUNDS, beta, errorMinStart):
    entryArr = np.array(entry)
    expectedExitArr = np.array(expectedExit)
    w = np.zeros(N+1)
    error = 1
    error_min = errorMinStart
    errors = []
    i = 0
    while error > 0 and i < MAX_ROUNDS:
        i_k = random.randint(0, K-1)
        h = np.dot(entryArr[i_k], w)
        g_h = g(h, beta)
        deltaW = eta*(expectedExitArr[i_k][0] - g_h)*entryArr[i_k]
        w = w + deltaW
        error = calculateErrorPandas(entryArr, expectedExitArr, w, g, K, beta)
        errors.append({'error': error, 'iteration': i})
        if error < error_min:
            error_min = error
            w_min = w
        i = i + 1
    return { 'errors': errors, 'w_min': w_min.tolist() }
