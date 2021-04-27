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
                                simpleDeltaWFunc, None, json_data['exercise'], json_data['error'], ex2Input, ex2DesiredOutput, 0.9)
    w_min = ans['w_min']
    errors = ans['errors']
    plotter.plotEx1(ans['w_min'], json_data['exitValues'])
    plotter.plotErrors(errors)
    plotter.plotAccuracy(ans['accuracyTraining'], ans['accuracyTesting'])
else:
    if json_data['exercise'] == 2:
        testingNumber = int(json_data['K']*0.1)
        trainingNumber = json_data['K'] - testingNumber
        testingArr = []
        testingOutputArr = []
        trainingArr = []
        trainingOutputArr = []
        for i in range(0, trainingNumber - 1):
            trainingArr.append(ex2Input[i])
            trainingOutputArr.append(ex2DesiredOutput[i])
        for i in range(trainingNumber, len(ex2Input)-1):
            testingArr.append(ex2Input[i])
            testingOutputArr.append(ex2DesiredOutput[i])
        ans = perceptron.simplePerceptron(trainingArr, trainingOutputArr, 
                                json_data['N'], json_data['K'], json_data['eta'], 
                                errorTypeFunc, simplePerceptronTypeFunc, json_data['limit'], json_data['beta'], json_data['errorMinStart'], 
                                simpleDeltaWFunc, simplePerceptronTypes.nonLinearDer, json_data['exercise'], json_data['error'], testingArr, testingOutputArr, 0.9)
        w_min = ans['w_min']
        errors = ans['errors']
        plotter.plotErrors(errors)
        plotter.plotAccuracy(ans['accuracyTraining'], ans['accuracyTesting'])
    else:
        if json_data['exercise'] == 3:
            p = MultiLayerPerceptron(json_data['alpha'], json_data['beta'], json_data['limit'], json_data['hiddenLayers'],
                json_data['error'], json_data['errorRange'], json_data['N'], json_data['isAdaptive'], 
                json_data['a'], json_data['b'])
            ans = p.train(json_data['function'])
            plotter.plotEx3Errors(ans['errorEpoch'], ans['wErrorEpoch'], ans['testEPE'], ans['testWorstEPE'])
            plotter.plotAccuracy(ans['accuracyTraining'], ans['accuracyTesting'])