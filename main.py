import perceptron
import calculateError
import simplePerceptronTypes
import json
import plotter
import simpleFileParser as sfp
import deltaW
import scaler
from multiLayerPerceptron import MultiLayerPerceptron 

with open('arguments.json', 'r') as j:
    json_data = json.load(j)

with open('config.json', 'r') as file:
    data = json.load(file)
    function = data['function']
    alpha = data['alpha']
    beta = data['beta']
    epochs = data['epochs']
    perceptron = data['perceptron']
    error = data['error']
    hiddenLayers = data['hiddenLayers']
    nodesPerLayer = data['nodesPerLayer']
    adaptive = data['adaptive']
    a = data['a']
    b = data['b']
    errorRange = data['errorRange']

if (perceptron == 'multi-layer'):
    p = MultiLayerPerceptron(alpha = alpha, beta = beta, iterations = epochs, hiddenLayers = hiddenLayers, error = error, errorRange = errorRange, nodesPerLayer = nodesPerLayer, adaptive = adaptive, a = a, b = b)

p.train(function)

"""

ex2Input = sfp.parseFile('ex2_input.tsv')
ex2DesiredOutput = sfp.parseFile('ex2_desired_output.tsv')
if json_data['exercise'] == 2 and json_data['simplePerceptronType'] == 'nonLinear':
    ex2Input = scaler.scale(ex2Input)
    ex2DesiredOutput = scaler.scale(ex2DesiredOutput)

switcherSimplePerceptronType = {
    'scalar': simplePerceptronTypes.scalar,
    'nonLinear' : simplePerceptronTypes.nonLinear,
    'linear': simplePerceptronTypes.linear,
}
simplePerceptronTypeFunc = switcherSimplePerceptronType.get(json_data['simplePerceptronType'], 'Invalid simple perceptron type')

switcherDeltaWCalculator = {
    'scalar': deltaW.linearW,
    'nonLinear': deltaW.nonLinearW,
    'linear': deltaW.linearW,
}
simpleDeltaWFunc = switcherDeltaWCalculator.get(json_data['simplePerceptronType'], 'Invalid simple perceptron type')

switcherErrorType = {
    'square': calculateError.squareError
}
errorTypeFunc = switcherErrorType.get(json_data['errorType'], 'Invalid error type')


if json_data['exercise'] == 1:
    ans = perceptron.simplePerceptron(json_data['entry'], json_data['exitValues'], 
                                json_data['N'], json_data['K'], json_data['eta'], 
                                errorTypeFunc, simplePerceptronTypeFunc, json_data['limit'], json_data['beta'], json_data['errorMinStart'],
                                simpleDeltaWFunc, None, json_data['exercise'])
    w_min = ans['w_min']
    errors = ans['errors']
    w_min[0] = json_data['entry'][0][0]*w_min[0]
    plotter.plotEx1(w_min, json_data['exitValues'])
    plotter.plotErrors(errors)
else:
    if json_data['exercise'] == 2:
        ans = perceptron.simplePerceptron(ex2Input, ex2DesiredOutput, 
                                json_data['N'], json_data['K'], json_data['eta'], 
                                errorTypeFunc, simplePerceptronTypeFunc, json_data['limit'], json_data['beta'], json_data['errorMinStart'], 
                                simpleDeltaWFunc, simplePerceptronTypes.nonLinearDer, json_data['exercise'])
        w_min = ans['w_min']
        errors = ans['errors']
        w_min[0] = ex2Input[0][0]*w_min[0]
        plotter.plotEx2(w_min, ex2Input)
        plotter.plotErrors(errors)
    else:
        if json_data['exercise'] == 3:
            i = 0
"""