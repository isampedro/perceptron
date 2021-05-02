from LinearTest import SimplePerceptron

from NoLinearTest import SimplePerceptronNoLinear

import os


import json

with open('input.json', 'r') as j:
    json_data = json.load(j)
    operand = json_data['FUNCTION']
    alpha = json_data['LEARNING_RATE']
    epochs = json_data['EPOCHS']
    perceptron = json_data['PERCEPTRON']
    beta = json_data['BETA']
    error_tolerance = json_data['ERROR_TOLERANCE']
    adaptive = json_data['ADAPTIVE_LEARNING_RATE']
    classification_margin = json_data['CLASSIFICATION_MARGIN']
    hidden_layers = json_data['HIDDEN_LAYERS']
    nodes_per_layer = json_data['NODES_PER_LAYER']
    adaptive = 'FALSE'
    k = json_data['K']

p = SimplePerceptron(eta=alpha, epochs=epochs, beta=beta, adaptive=adaptive, k=k)

p.crossValidation(operand)