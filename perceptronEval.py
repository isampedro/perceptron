import perceptron
import calculateError
import simplePerceptronTypes
import json
import plotter

with open('arguments.json', 'r') as j:
    json_data = json.load(j)

switcherSimplePerceptronType = {
    'scalar': simplePerceptronTypes.scalar,
}
simplePerceptronTypeFunc = switcherSimplePerceptronType.get(json_data['simplePerceptronType'], 'Invalid simple perceptron type')

switcherErrorType = {
    'square': calculateError.squareError
}
errorTypeFunc = switcherErrorType.get(json_data['errorType'], 'Invalid error type')

ans = perceptron.simplePerceptron(json_data['entry'], json_data['exitValue'], 
                                    json_data['N'], json_data['K'], json_data['eta'], 
                                    errorTypeFunc, simplePerceptronTypeFunc, json_data['limit'])

plotter.plot(ans)