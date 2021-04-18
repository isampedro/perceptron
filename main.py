import perceptron
import calculateError
import simplePerceptronTypes
import json
import plotter
import simpleFileParser as sfp
import deltaW
import scaler

with open('arguments.json', 'r') as j:
    json_data = json.load(j)

ex2Input = sfp.parseFile('ex2_input.tsv')
ex2Input = scaler.scale(ex2Input)
for elem in ex2Input:
    elem.insert(0, 1)
ex2DesiredOutput = sfp.parseFile('ex2_desired_output.tsv')
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
                                errorTypeFunc, simplePerceptronTypeFunc, json_data['limit'], json_data['beta'], json_data['errorMinStart'], simpleDeltaWFunc, None)
    w_min = ans['w_min']
    errors = ans['errors']
    w_min[0] = json_data['entry'][0][0]*w_min[0]
    plotter.plotEx1(w_min, json_data['exitValues'])
    plotter.plotErrors(errors)
else:
    if json_data['exercise'] == 2:
        ans = perceptron.simplePerceptron(ex2Input, ex2DesiredOutput, 
                                json_data['N'], json_data['K'], json_data['eta'], 
                                errorTypeFunc, simplePerceptronTypeFunc, json_data['limit'], json_data['beta'], json_data['errorMinStart'], simpleDeltaWFunc, simplePerceptronTypes.nonLinearDer)
        w_min = ans['w_min']
        errors = ans['errors']
        w_min[0] = ex2Input[0][0]*w_min[0]
        plotter.plotEx2(w_min)
        plotter.plotErrors(errors)
    else:
        if json_data['exercise'] == 3:
            i = 0
