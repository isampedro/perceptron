import numpy as np
import plotter



# El índice i se refiere a las unidades de salida
# El índice j se refiere a las unidades ocultas
# El k se refiere a las unidades de entrada
# El u indica el número de ejemplo y varía entre 1 y p donde p es la cantidad de ejemplos en el conjunto de entrenamiento
# V --> estado de activación de las unidades de la capa oculta --> bidimensional [nro de capa][nro de nodo]
# W --> conexiones entre la capas --> tridimensional [capa destino, nodo destino, nodo origen]
# w --> conexiones entre los nodos de entrada y la capa oculta --> bidimensional [nodo destino, nodo origen]


class MultiLayerPerceptron:
    
    def __init__(self, alpha, beta, iterations, hiddenLayers, error, errorRange, nodesPerLayer, adaptive, a, b, trainingSize):
        self.alpha = alpha 
        self.beta = beta 
        self.iterations = iterations # máxima cantidad de épocas que puede tener el algoritmo
        self.hiddenLayers = hiddenLayers # cantidad de capas ocultas
        self.totalLayers = hiddenLayers + 2 
        self.nodesPerLayer = nodesPerLayer # cantidad de nodos que tengo en una capa
        self.error = error  # error que tolero
        self.errorRange = errorRange    # margen de error tolerado 
        self.adaptive = adaptive
        self.a = a
        self.b = b
        self.trainingSize = trainingSize
        self.readContent = []



    def readFile(self, size, test):
        if test == True:
            X = self.readEj(5,7,10,test).tolist()
            Y = self.readContent.tolist()
            if len(Y) < 10:
                for elem in Y:
                    X.remove(elem)
            return X
        if test != True:
            return self.readEj(5,7,10,test)

    def readEj(self, width, height, amount, test):
        f = open('mapa-de-pixeles.txt', 'r')
        linesf = f.read().split('\n')
        valuesf = [line.strip() for line in linesf]
        values_indiv = [line.split(' ') for line in valuesf]
        data = np.zeros((10, width*height + 2))
        data2 = np.zeros((amount, width*height + 2))
        for index in range(10):
            for fila in range(height):
                for col in range(width):
                    data[index][1+col+fila*width] = int(values_indiv[index*height+fila][col])
        for i in range(len(data)):
            data[i][0] = 1
            data[i][-1] = (i%2 * 2) - 1
        np.random.shuffle(data)
        data2 = data[0:amount]
        if not test:
            self.readContent = data2
        return data2


    # g() es la función que utilizo --> g(h) = tangh(beta * h)
    def g(self, x):
        return np.tanh(self.beta * x)
    # función derivada de g() --> g´(x) = 1 / cosh^2 (x) 
    def derivatedG (self, x):
        cosh2 = (np.cosh(self.beta * x)) ** 2
        return self.beta / cosh2
    # función para calcular el estado de excitación del nodo 
    def h(self, m, i, nodes, W, V):
        hmi = 0
        for j in range(0, nodes):
            hmi += W[m, i, j] * V[m-1][j]
        return hmi

    def adaptiveAlpha(self, errors):
        if(len(errors) > 8):
            lastErros = errors[-8:]
            aux = []
            aux2 = []
            for i in range(len(lastErros) - 1):
                aux.append(lastErros[i] < lastErros[i+1])
            for i in range(len(lastErros) - 1):
                aux2.append(lastErros[i] > lastErros[i+1])
            if all(aux):
                self.alpha -= self.b * self.alpha
            if all(aux2):
                self.alpha += self.a
            
            

    # ALGORITMO (slide 30)

    def train(self, ej):
        if ej == "XOR":
        #           bias   x    y    salida
            data = [[1.0, 1.0, 1.0, -1.0],
                    [1.0, -1.0, 1.0, 1.0],
                    [1.0, 1.0, -1.0, 1.0],
                    [1.0, -1.0, -1.0, -1.0]]
        
        if ej == "EVEN":
            data = self.readFile(self.trainingSize, False)

        errorEpoch = []
        wErrorEpoch = []
        accuracy = []

        errorMin = len(data) * 2

        if ej == "XOR":         
            test_data = [[1.0,  1.0,  1.0, -1.0],
                        [1.0, -1.0,  1.0,  1.0],
                        [1.0,  1.0, -1.0,  1.0],
                        [1.0, -1.0, -1.0, -1.0]]
        if ej == "EVEN":
            test_data = self.readFile(self.trainingSize, True)
        test_error_per_epoch = []
        test_worst_error_per_epoch = []
        test_accuracy = []
        
        
        # M es el índice de la capa superior 
        self.M = self.totalLayers - 1
        # nodos en la capa superior
        self.exitNodes = 1
        # inicializo el estado de activación de todas las unidades en la capa oculta --> [capa, nro de nodo]
        self.V = np.zeros((self.M + 1, self.nodesPerLayer))
        # el estado de activación de la primera unidad de todas las capas ocultas deben tener el mismo valor --> 1 en este caso (BIAS)
        for i in range(1, self.M):
            self.V[i][0] = 1

        # inicializar el conjunto de pesos en valores pequeños al azar (PASO 1)

        # inicializo las conexiones entre capas [capa destino, nodo destino, nodo origen]
        self.W = np.random.rand(self.M+1, self.nodesPerLayer, self.nodesPerLayer)
        #  conexiones de los nodos de entrada y la capa oculta [nodo destino, nodo origen]
        w = np.random.rand(self.nodesPerLayer, len(data[0]))
        self.W[1,:,:] = np.zeros((self.nodesPerLayer, self.nodesPerLayer))
        # inicializo en 0 los errores en las capas 
        self.d = np.zeros((self.M + 1, self.nodesPerLayer))
        # creo las conexiones de los nodos de entrada y la capa oculta
        for origen in range(len(data[0]) - 1):
            for destinatario in range(self.nodesPerLayer):
                self.W[1,destinatario,origen] = w[destinatario,origen] # trabajo todo en la capa 1
        
        for epoch in range(1, self.iterations):
            totalError = 0
            wError = 0
            corrects = 0
            incorrects = 0
            # tomar un ejemplo al azar del conjunto de entrenamiento y aplicarlo a la capa 0 (PASO 2)
            np.random.shuffle(data) # mezclo el conjunto de entrenamiento al azar para seleccionar el primero
            # tomo un ejemplo (u)
            for u in range(len(data)):
                for k in range(len(data[0]) - 1):
                    #nodos de entrada
                    self.V[0][k] = data[u][k]
                # propagar la entrada hasta la capa de salida (PASO 3)
                for m in range(1, self.M):
                    for j in range(1, self.nodesPerLayer):
                        hmj = self.h(m, j, self.nodesPerLayer, self.W, self.V)
                        self.V[m][j] = self.g(hmj)
                # en la última capa hay distinta cantidad de nodos  (PASO 3)
                for i in range(0, self.exitNodes):
                    hmi = self.h(self.M, i, self.nodesPerLayer, self.W, self.V)
                    self.V[self.M][i] = self.g(hmi)
                if self.V[self.M][i] >= data[u][-1] - self.errorRange and self.V[self.M][i] <= data[u][-1] + self.errorRange:
                    corrects += 1
                else:
                    incorrects += 1

                # calcular el error para la capa de salida (PASO 4)
                for i in range(0, self.exitNodes):
                    #  n[-1] es el último item de la lista n --> data[u][-1] es la salida deseada de mi ejemplo tomado
                    hmi = self.h(self.M, i, self.nodesPerLayer, self.W, self.V)
                    self.d[self.M][i] = self.derivatedG(hmi) * (data[u][-1] - self.V[self.M][i])
                # retropropagar el error (PASO 5)  
                for m in range(self.M, 1, -1):  #range(start_value, end_value, step)
                    for j in range(0, self.nodesPerLayer):
                        hpmi = self.h(m-1, j, self.nodesPerLayer, self.W, self.V)
                        errorSum = 0
                        for i in range(0, self.nodesPerLayer):
                            errorSum += self.W[m,i,j] * self.d[m][i]
                        self.d[m-1][j] = self.derivatedG(hpmi) * errorSum
                # actualizar los pesos de las conexiones (PASO 6)
                for m in range(1, self.M + 1):
                    for i in range(self.nodesPerLayer):
                        for j in range(self.nodesPerLayer):
                            deltaW = self.alpha * self.d[m][i] * self.V[m-1][j]
                            self.W[m,i,j] = self.W[m,i,j] + deltaW
                #Calcular el error. Si error > COTA, ir al paso 2 (PASO 7)
                for i in range(0, self.exitNodes):
                    # comparo la salida deseada con la salida de la unidad de la última capa
                    if abs(data[u][-1] - self.V[self.M][i]) > wError:
                        wError = abs(data[u][-1] - self.V[self.M][i])
                    totalError += abs(data[u][-1] - self.V[self.M][i])
                    
                
            errorEpoch.append(totalError/len(data))
            wErrorEpoch.append(wError)
            # utilizo el 0.0 para pasar corrects e incorrects (que son enteros) a float
            accuracy.append (corrects / (0.0 + corrects  + incorrects ))

            if self.adaptive and epoch % 8 == 0:
                self.adaptiveAlpha(errorEpoch)

            if totalError < errorMin:
                errorMin = totalError
                self.w_min = self.W
                
            if totalError <= self.error*len(data) or epoch == self.iterations - 1:
                self.makeTest(test_data, self.w_min, epoch, test_error_per_epoch, test_worst_error_per_epoch, test_accuracy)
                break
            else:
                self.makeTest(test_data, self.w_min, epoch, test_error_per_epoch, test_worst_error_per_epoch, test_accuracy)
        return { 'errorEpoch': errorEpoch, 'wErrorEpoch': wErrorEpoch, 'accuracy': accuracy }


def makeTest(self, test_data, weights, epoch, test_error, test_worst_error, test_accuracy):
    element_count = 0
    total_error = 0
    worst_error = 0
    W = weights
    positives = 0
    negatives = 0
    for row in test_data:
        for k in range(len(row)-1):
            self.V[0][k] = row[k]
        for m in range(1, self.M):
            for i in range(1, self.nodesPerLayer):
                hmi = self.h(m,i,self.nodesPerLayer,W,self.V)
                self.V[m][i] = self.g(hmi)
        for i in range(0, self.exitNodes):
            hmi = self.h(self.M,i,self.nodesPerLayer,W,self.V)
            self.V[self.M][i] = self.g(hmi)
        perceptron_output = self.V[self.M][0]
        element_count += 1
        if perceptron_output >= row[-1] - self.errorRange and perceptron_output <= row[-1] + self.errorRange:
            positives += 1
        else:
            negatives +=1
        if abs(perceptron_output - row[-1]) > worst_error:
            worst_error = abs(perceptron_output - row[1])
    total_error.append(total_error / len(test_data))
    test_worst_error.append(worst_error)
    test_accuracy.append(positives / (0.0 + positives + negatives))
        
    

        
                    
                 

