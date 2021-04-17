import perceptron
import calculateError
import simplePerceptronTypes
import json
import plotter
import simpleFileParser as sfp

with open('arguments.json', 'r') as j:
    json_data = json.load(j)

ex2Input = sfp.parseFile('ex2_input.tsv')
ex2DesiredOutput = sfp.parseFile('ex2_desired_output.tsv')

switcherSimplePerceptronType = {
    'scalar': simplePerceptronTypes.scalar,
    'nonLinear' : simplePerceptronTypes.nonLinear,
}
simplePerceptronTypeFunc = switcherSimplePerceptronType.get(json_data['simplePerceptronType'], 'Invalid simple perceptron type')


if json_data['exercise'] == 1:
    switcherErrorType = {
        'square': calculateError.squareErrorJSON
    }
    errorTypeFunc = switcherErrorType.get(json_data['errorType'], 'Invalid error type')

    ans = perceptron.simplePerceptronJSON(json_data['entry'], json_data['exitValues'], 
                                json_data['N'], json_data['K'], json_data['eta'], 
                                errorTypeFunc, simplePerceptronTypeFunc, json_data['limit'], json_data['beta'])
    w_min = ans['w_min']
    errors = ans['errors']
    w_min[0] = json_data['entry'][0][0]*w_min[0]
    plotter.plotEx1(w_min, json_data['exitValues'])
    plotter.plotErrors(errors)
else:
    if json_data['exercise'] == 2:
        switcherErrorType = {
            'square': calculateError.squareErrorPandas
        }
        errorTypeFunc = switcherErrorType.get(json_data['errorType'], 'Invalid error type')
        ans = perceptron.simplePerceptronPandas(ex2Input, ex2DesiredOutput, 
                                json_data['N'], json_data['K'], json_data['eta'], 
                                errorTypeFunc, simplePerceptronTypeFunc, json_data['limit'], json_data['beta'])
        w_min = ans['w_min']
        errors = ans['errors']
        w_min[0] = ex2Input[0][0]*w_min[0]
        plotter.plotEx2(w_min)
        plotter.plotErrors(errors)
