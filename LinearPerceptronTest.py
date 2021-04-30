from matplotlib.lines import Line2D
import numpy as np
import json

input_data = np.genfromtxt("./data/train_set.txt", delimiter=",")
input_norm = np.linalg.norm(input_data)
input_data = input_data / input_norm

output_data =  np.genfromtxt("./data/expected_outputs.txt", delimiter=" ")
output_norm = np.linalg.norm(output_data)
output_data = output_data / output_norm

def fit(initial_weights, learn_factor, inputs, expected_values, limit):
    samples = inputs.shape[0] # me devulve la cantidad de filas del arreglo
    epochs = 0
    new_weights = initial_weights
    # agrego el bias 
    x = np.concatenate([inputs, np.ones((samples, 1))], axis=1)
    count = 0
    for i in range(limit):
        error = 0
        for j in range(samples):
            theta = np.dot(new_weights, x[j, :])
            deltaW = learn_factor * (expected_values[j] - theta) * x[j, :]
            new_weights += deltaW
            error += np.power(expected_values[j] - theta, 2) / 2
        epochs += 1

        if i == limit - 1:
            print ("Epochs: ", epochs)
            print(samples)
            print("Train error: ", error / samples)
            return new_weights, error, epochs

def predict(inputs, expected_values, weights):
    samples = inputs.shape[0]
    # agrego el bias
    x = np.concatenate([inputs, np.ones((samples, 1))], axis = 1)
    outputs = np.array([])
    error = 0
    for i in range(len(x)):
        theta = np.dot(weights, x[i])
        error += np.power(expected_values[i] - theta, 2) / 2
        outputs = np.append(outputs, theta)
    print(samples)
    error = error / samples
    return outputs, error


with open('./data/config.json') as json_file:
    data = json.load(json_file)
    for p in data['ej2']:
        total_epochs = int(p['total_epochs'])
        epoch_step = int(p['epoch_step'])
        # donde arranco, fin del intervalo (el intervalo no incluye este valor), espacio entre valores
        epoch_array = np.arange(epoch_step, total_epochs + epoch_step, epoch_step)
        learn_factor = float(p['learn_factor'])
        k = int(p['k'])
        cross_validation = (p['cross_validation'])
        

# Divido en conjunto de prueba y de entrenamiento
test_size = int(len(input_data) / k)
train_size = len(input_data) - test_size
# creo los indices de los conjuntos
indexes = np.arange(0, len(input_data))
# randomizo los indices
np.random.shuffle(indexes)
# tomo los indices para el conjunto de prueba
test_indexes = np.split(indexes, k)
test_set = np.empty((0,3), float)
train_sets = np.empty((0,3), float)
test_outputs = np.array([])
train_outputs = np.array([])


# cargo los datos en mi conjunto de entrenamiento y de prueba
for i in range(k):
    index = test_indexes[i]
    test_set = np.append(test_set, np.take(input_data, index, 0), axis = 0 )
    test_outputs = np.append(test_outputs, np.take(output_data, index, 0))
    train_sets = np.append(train_sets, np.delete(input_data, index, 0), axis = 0 )
    train_outputs = np.append(train_outputs, np.delete(output_data, index, 0), axis = 0)

# chequear si es necesario hacerlo para el conjunto de prueba
test_set = np.split(test_set, k)
test_outputs = np.split(test_outputs, k)
train_sets = np.split(train_sets, k)
train_outputs = np.split(train_outputs, k)

# corro el algoritmo neuronal
weights = np.random.random_sample(input_data.shape[1] + 1) * 2 -1    
train_set_error_history = np.array([])
test_set_error_history = np.array([])
epochs_history = np.array([])
final_output = []
min_weights = []
min_index = 0
min_test_error = 100000
min_train_error = 0
min_test_output = []

if cross_validation == "false":
    k = 1
for epoch in epoch_array:
    epochs_history = np.append(epochs_history, epoch)
    for i in range(k):
        print("----------------------------------------")
        print("K = ", i)
        new_weights, train_error, epochs = fit(weights, learn_factor, train_sets[i], train_outputs[i], epoch)
        final_output, test_error = predict(test_set[i], test_outputs[i], weights)
        print("Test Error: ", test_error)
        if test_error < min_test_error:
            min_weights = new_weights
            min_train_error = train_error
            min_test_error = test_error
            min_test_output = final_output
            min_index = i
    weights = min_weights
    print("Minimum test error: ", min_test_error)












            







        



