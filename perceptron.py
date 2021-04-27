import numpy as np
import random
import json
import calculateError as cError

def simplePerceptron(entry, expectedExit, N, K, eta, calculateError, g, MAX_ROUNDS, beta, errorMinStart, deltaWFunc, 
    gder, exercise, deltaError, entryTesting, expectedExitTesting, trainingQ, isAdaptive, a, b):
    for elem in entry:
        elem.insert(0, 1)
    for elem in entryTesting:
        elem.insert(0, 1)
    entryArr = np.array(entry)
    entryTestingArr = np.array(entryTesting)
    np.random.shuffle(entryTestingArr)
    expectedExitArr = np.array(expectedExit)
    shuffler = np.random.permutation(len(entryArr))
    entryArr = entryArr[shuffler]
    expectedExitArr = expectedExitArr[shuffler]
    expectedExitTestingArr = np.array(expectedExitTesting)
    np.random.shuffle(expectedExitTestingArr)
    w = np.zeros(N+1)
    error = 1
    epochs = 0
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
        error = calculateError(entryTestingArr, expectedExitTestingArr, w, g, K - int(K*trainingQ), beta)
        errorsTesting.append({'error': error, 'iteration': i})
        accuracy = cError.accuracy(entryArr, expectedExitArr, w, g, int(K*trainingQ), beta, deltaError)
        accuracies.append(accuracy)
        accuracy = cError.accuracy(entryTestingArr, expectedExitTestingArr, w, g, K - int(K*trainingQ), beta, deltaError)
        accuraciesTesting.append(accuracy)
        if error < error_min:
            error_min = error
            w_min = w
        i = i + 1
        epochs += 1
        if epochs % 8 == 0 and isAdaptive:
            eta = adaptiveAlpha(errors, eta, a, b)
    return { 'errors': errors, 'errorsTesting': errorsTesting, 'w_min': w_min.tolist(), 'accuracyTesting': accuraciesTesting, 'accuracyTraining': accuracies, 'finalEta': eta }

def adaptiveAlpha(errors, eta, a, b):
    if(len(errors) > 8):
        lastErros = errors[-8:]
        aux = []
        aux2 = []
        for i in range(len(lastErros) - 1):
            aux.append(lastErros[i]['error'] < lastErros[i+1]['error'])
        for i in range(len(lastErros) - 1):
            aux2.append(lastErros[i]['error'] > lastErros[i+1]['error'])
        if all(aux):
            eta -= b*eta
        if all(aux2):
            eta += a
    return eta

def crossValidation(inputArr, expectedOutput, N, K, eta, calculateError, g, MAX_ROUNDS, beta, errorMinStart, deltaWFunc, gder, exercise,
                    deltaError, isAdaptive, a, b, elems):
    k = int(K / elems)
    #SPLITEO LOS DATOS EN K GRUPOS
    inputGroups = np.array_split(inputArr, k)
    expectedOutputGroups = np.array_split(expectedOutput, k)
    trainingInputList = []
    trainingExpectedOutputList = []
    answers = []

    #ITERO POR CADA GRUPO, MANTENIENDO LA i DE PIVOTE
    for i in range(k):
        trainingInputList = []
        trainingExpectedOutputList = []
        #SETEO AL RESTO COMO SETS DE TRAINING
        for j in range(k):
            if j != i:
                trainingInputList.extend(inputGroups[j].tolist())
                trainingExpectedOutputList.extend(expectedOutputGroups[j].tolist())

        #ITERO HASTA MAX_ROUNDS
        ans = simplePerceptron(trainingInputList.copy(), trainingExpectedOutputList.copy(), N, k, eta, calculateError, g, MAX_ROUNDS, beta, errorMinStart, deltaWFunc,
            gder, exercise, deltaError, inputGroups[i].tolist().copy(), expectedOutputGroups[i].tolist().copy(), 0.9, isAdaptive, a, b)
    return ans