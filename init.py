from LinearTest import SimplePerceptron
from NoLinearTest import SimplePerceptronNoLinear
from MultiP import MultiLayerPerceptron
from file_reader import Reader
import numpy as np
import os


import json

with open('input.json', 'r') as j:
    json_data = json.load(j)
    perceptron = json_data['PERCEPTRON']
    operand = json_data['FUNCTION']
    eta = json_data['LEARNING_RATE']
    training_size = json_data['TRAINING_SIZE']
    epochs = json_data['EPOCHS']
    error_tolerance = json_data['ERROR_TOLERANCE']
    momentum = json_data['MOMENTUM']
    adaptive_eta = json_data['ADAPTIVE_LEARNING_RATE']
    adaptive_eta_increase = json_data['ADAPTIVE_ETA_INCREASE']
    adaptive_eta_decrease = json_data['ADAPTIVE_ETA_DECREASE']
    beta = json_data['BETA']
    delta_accuracy_error = json_data['DELTA_ACCURACY_ERROR']
    hidden_layers = json_data['HIDDEN_LAYERS']
    nodes_per_layer = json_data['NODES_PER_LAYER']
    cross_validation = json_data['CROSS_VALIDATION']
    k = json_data['K']

if cross_validation == "FALSE":
    cross_validation = False
else:
    cross_validation = True

if adaptive_eta == "FALSE":
    adaptive_eta  = False
else:
    adaptive_eta  = True

if momentum == "FALSE":
    momentum  = False
else:
    momentum  = True
    

p = MultiLayerPerceptron(eta, beta, epochs, hidden_layers, nodes_per_layer,error_tolerance, adaptive_eta, delta_accuracy_error, training_size, momentum, adaptive_eta_increase, adaptive_eta_decrease, cross_validation, k)

if cross_validation:
    r = Reader('Ej3')
    train_data, test_data = r.readFile(training_size, cross_validation, k)
    data = train_data
    last_partition = 0
    split_number = 10 / k
    for i in range(1, k):
        partition = i * split_number
        test_data = data[int(last_partition):int(partition)]
        train_data1 = data[int(last_partition):]
        train_data2 = data[int(partition):]
        train_data = np.concatenate((train_data1, train_data2)) 
        last_partition = partition
        p.algorithm_cross_validation("EVEN", train_data, test_data)
    else:
        p.algorithm("EVEN") 