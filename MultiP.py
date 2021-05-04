from matplotlib import pyplot as plt 
import numpy as np
import random
from file_reader import Reader
from plotter import Plotter

class MultiLayerPerceptron:

    def __init__(self, eta, beta, epochs, hidden_layers, nodes_per_layer, error_tolerance, adaptive_eta, delta_accuracy_error, training_set_size, momentum, adaptive_eta_increase, adaptive_eta_decrease, cross_validation, k):
        self.eta = eta
        self.initial_eta = eta
        self.beta = beta # usado para la función de activación 
        self.delta_accuracy_error = delta_accuracy_error 
        self.epochs = epochs
        self.adaptive = adaptive_eta
        self.hidden_layers = hidden_layers
        self.total_layers = hidden_layers + 2 # input + hidden + output
        self.error_tolerance = error_tolerance
        self.nodes_per_layer = nodes_per_layer
        self.training_set_size = training_set_size
        self.momentum = momentum
        self.adaptive_eta_increase = adaptive_eta_increase
        self.adaptive_eta_decrease = adaptive_eta_decrease
        self.last_delta_w = 0
        self.cross_validation = cross_validation
        self.k = k

    def adjust_learning_rate(self, errors_so_far):
        if(len(errors_so_far) > 10):
            last_10_errors = errors_so_far[-10:]
            booleans = []
            for i in range(len(last_10_errors) - 1):
                booleans.append(last_10_errors[i] > last_10_errors[i + 1])
            if all(booleans):
                self.eta += self.adaptive_eta_increase
            else:
                self.eta -= self.adaptive_eta_decrease * self.eta

    # g(h) = tanh(βh)
    def g(self, x):
        return np.tanh(self.beta * x)

    # La derivada de la tangente hiperbólica es el inverso del coseno hiperbólico al cuadrado de x.
    # --> derivada de tanh(x) = 1 / cosh^2(x)
    def g_derivative(self, x):
        cosh2 = (np.cosh(self.beta*x)) ** 2
        return self.beta / cosh2
    
    # Cada unidad de salida alcanza un estado de exitación hµi dado po hµi = sumatoriaj(Wij * Vµj)
    def h(self, m, i, amount_of_nodes, W, V):
        hmi = 0
        for j in range(0, amount_of_nodes):
            hmi += W[m,i,j] * V[m-1][j]
        return hmi

    # Vmi = g(sumatoriaj( wmij * Vm−1j) para todo m desde 1 hasta M.
    def propagate(self):
        for m in range(1, self.M):
            for i in range(1, self.nodes_per_layer):
                hmi = self.h(m, i, self.nodes_per_layer, self.W, self.V)
                self.V[m][i] = self.g(hmi)

    # δm−1i = derivada de g(hm−1i) * (sumatoriaj(wmji * δmj)) para todo m entre M y 2
    def back_propagate(self):
        for m in range(self.M, 1 ,-1): 
            #range(start_value, end_value, step)                                          
            for j in range(0, self.nodes_per_layer):                             
                hprevmi = self.h(m-1, j, self.nodes_per_layer, self.W, self.V)   
                error_sum = 0
                for i in range(0, self.nodes_per_layer):                         
                    error_sum += self.W[m,i,j] * self.d[m][i]                    
                self.d[m-1][j] = self.g_derivative(hprevmi) * error_sum
        return error_sum
    
    # w.nuevomij = w.viejomij + ∆wmij donde ∆wmij = ηδmi * Vjm−1
    def update_weights(self):
        for m in range(1, self.M+1):
            for i in range(self.nodes_per_layer):
                for j in range(self.nodes_per_layer):
                    delta_w = self.eta * self.d[m][i] * self.V[m-1][j]
                    if self.momentum:
                        delta_w +=  0.8 * self.last_delta_w
                    self.W[m,i,j] += delta_w
                    self.last_delta_w = delta_w
        
    def algorithm(self, problem):
        #                           bias    x     y    out
        if problem == "XOR": 
            train_data = [[1.0,  1.0,  1.0, -1.0],
                    [1.0, -1.0,  1.0,  1.0],
                    [1.0,  1.0, -1.0,  1.0],
                    [1.0, -1.0, -1.0, -1.0]]
            test_data = [[1.0,  1.0,  1.0, -1.0],
                        [1.0, -1.0,  1.0,  1.0],
                        [1.0,  1.0, -1.0,  1.0],
                        [1.0, -1.0, -1.0, -1.0]]

        if problem == "EVEN":
            r = Reader('Ej3')
            train_data, test_data = r.readFile(self.training_set_size, self.cross_validation, self.k)
          
        # M es el índice de la capa superior                       
        self.M = self.total_layers - 1      
        # cuantos nodos hay en la capa oculta (incluyendo el bias)                            
        self.nodes_per_layer = max(self.nodes_per_layer, len(train_data[0]) - 1)
        # nodos en la capa superior                 
        self.exit_nodes = 1     
        # inicializo el estado de activación de todas las unidades en la capa oculta --> [capa, nro de nodo]                                        
        self.V = np.zeros((self.M + 1, self.nodes_per_layer))
         # el estado de activación de la primera unidad de todas las capas ocultas deben tener el mismo valor --> 1 en este caso (BIAS)           
        for i in range(1, self.M):
            self.V[i][0] = 1
        # PASO 1: inicializar el conjunto de pesos en valores pequeños al azar                                                                
        self.W = np.random.rand(self.M+1, self.nodes_per_layer, self.nodes_per_layer)-0.5   # [capa destino, dest, origen]
        w = np.random.rand(self.nodes_per_layer, len(train_data[0]) - 1)-0.5                      # [dest, origen]
        self.W[1,:,:] = np.zeros((self.nodes_per_layer, self.nodes_per_layer))


        # inicializo en 0 los errores en las capas 
        self.d = np.zeros((self.M+1, self.nodes_per_layer))
        # creo las conexiones de los nodos de entrada y la capa oculta
        for nodo_origen in range(len(train_data[0]) - 1):
            for nodo_destino in range(self.nodes_per_layer):
                # capa destino, nodo destino, nodo origen
                self.W[1,nodo_destino,nodo_origen] = w[nodo_destino, nodo_origen]
        
        error_min = 1000000
        delta_accuracy_error = self.delta_accuracy_error 
        total_error = 1
        train_error_per_epoch = []
        test_error_per_epoch = []
        train_worst_error_per_epoch = []
        test_worst_error_per_epoch = []
        train_accuracy = []
        test_accuracy = []
        plotter = Plotter()
        
        for epoch in range(1, self.epochs):
            total_error = 0
            train_worst_error_this_epoch = 0
            corrects = 0
            incorrects = 0
            # mezclo el conjunto de entrenamiento al azar para seleccionar el primero
            np.random.shuffle(train_data)
            # tomo un ejemplo (u)
            for u in range(len(train_data)):
                # Paso 2 (V0 tiene los ejemplos iniciales)
                for k in range(len(train_data[0])-1):
                    # nodos de entrada
                    self.V[0][k] = train_data[u][k]
                # PASO 3: propagar la entrada hasta la capa de salida
                self.propagate()
                # en la última capa hay distinta cantidad de nodos 
                for i in range(0, self.exit_nodes):
                    hMi = self.h(self.M, i, self.nodes_per_layer, self.W, self.V)
                    self.V[self.M][i] = self.g(hMi)
                if self.V[self.M][i] >= train_data[u][-1] - delta_accuracy_error and self.V[self.M][i] <= train_data[u][-1] + delta_accuracy_error:
                    corrects += 1
                else:
                    incorrects += 1

                # PASO 4: calcular el error para la capa de salida
                for i in range(0, self.exit_nodes):
                    #  n[-1] es el último item de la lista n --> data[u][-1] es la salida deseada de mi ejemplo tomado
                    hMi = self.h(self.M, i, self.nodes_per_layer, self.W, self.V)
                    self.d[self.M][i] = self.g_derivative(hMi)*(train_data[u][-1] - self.V[self.M][i])
                # PASO 5: retropropagar el error 
                error_sum = self.back_propagate()
                # PASO 6: actualizar los pesos de las conexiones
                self.update_weights()
                # PASO 7: Calcular el error. Si error > COTA, ir al paso 2
                for i in range(0, self.exit_nodes):
                    if abs(train_data[u][-1] - self.V[self.M][i]) > train_worst_error_this_epoch:
                        train_worst_error_this_epoch = abs(train_data[u][-1] - self.V[self.M][i])
                    total_error += abs(train_data[u][-1] - self.V[self.M][i])
            train_error_per_epoch.append(total_error/len(train_data))
            train_worst_error_per_epoch.append(train_worst_error_this_epoch)
            train_accuracy.append(corrects / (0.0 + corrects + incorrects))
            if self.adaptive and epoch % 10 == 0:
                self.adjust_learning_rate(train_error_per_epoch)
            if total_error < error_min:
                error_min = total_error
                self.w_min = self.W
            if total_error/len(train_data) <= self.error_tolerance or epoch == self.epochs-1:
                self.test_perceptron(test_data, self.w_min, epoch, test_error_per_epoch, test_worst_error_per_epoch, test_accuracy, delta_accuracy_error, True, train_accuracy, train_error_per_epoch)
                break
            else:
                self.test_perceptron(test_data, self.w_min, epoch, test_error_per_epoch, test_worst_error_per_epoch, test_accuracy, delta_accuracy_error, False, train_accuracy, train_error_per_epoch)
        plotter.ej3_errors(train_error_per_epoch, test_error_per_epoch)
        plotter.ej3_accuracy(train_accuracy, test_accuracy)
        return

    

    def test_perceptron(self, test_data, weights, epoch, test_error, test_worst_error, test_accuracy, delta_accuracy_error, printing, train_accuracy, train_error):
        element_count = 0
        if printing:
            print("Testing perceptron for epoch %d..." %(epoch+1))
            print('+-------------------+-------------------+')
            print('|   Desired output  |   Perceptron out  |')
            print('+-------------------+-------------------+')
        total_error = 0
        worst_error = 0
        corrects = 0
        incorrects = 0
        W = weights
        for row in test_data:
            for k in range(len(row)-1):
                self.V[0][k] = row[k]
            for m in range(1, self.M):
                for i in range(1, self.nodes_per_layer):
                    hmi = self.h(m, i, self.nodes_per_layer, W, self.V)
                    self.V[m][i] = self.g(hmi)
            for i in range(0, self.exit_nodes):
                hMi = self.h(self.M, i, self.nodes_per_layer, W, self.V)
                self.V[self.M][i] = self.g(hMi)
            perceptron_output = self.V[self.M][0]
            element_count += 1
            if printing:
                print('       {}\t    |  {}'.format(row[-1], perceptron_output))
            if perceptron_output >= row[-1] - delta_accuracy_error and perceptron_output <= row[-1] + delta_accuracy_error:
                corrects += 1
            else:
                incorrects += 1
            if abs(perceptron_output- row[-1]) > worst_error:
                worst_error = abs(perceptron_output- row[-1])
            total_error += abs(perceptron_output- row[-1])
        test_error.append(total_error/len(test_data))
        test_worst_error.append(worst_error)
        test_accuracy.append(corrects / (0.0 + corrects + incorrects))
        if printing:
            print('Analysis finished for epoch %d' %(epoch+1))
            print('Final test accuracy: {}'.format(test_accuracy[len(test_accuracy)-1]))
            print('Final test error: {}'.format(test_error[len(test_error)-1]))
            print('Final train accuracy: {}'.format(train_accuracy[len(train_accuracy)-1]))
            print('Final train error: {}'.format(train_error[len(train_error)-1]))
            print('Initial learning rate: {}'.format(self.initial_eta))
            print('End learning rate: {}'.format(self.eta))
            print('+-------------------+-------------------+')
            print('Test finished')



    def algorithm_cross_validation(self, problem, train_data, test_data):
        # M es el índice de la capa superior                       
        self.M = self.total_layers - 1      
        # cuantos nodos hay en la capa oculta (incluyendo el bias)                            
        self.nodes_per_layer = max(self.nodes_per_layer, len(train_data[0]) - 1)
        # nodos en la capa superior                 
        self.exit_nodes = 1     
        # inicializo el estado de activación de todas las unidades en la capa oculta --> [capa, nro de nodo]                                        
        self.V = np.zeros((self.M + 1, self.nodes_per_layer))
         # el estado de activación de la primera unidad de todas las capas ocultas deben tener el mismo valor --> 1 en este caso (BIAS)           
        for i in range(1, self.M):
            self.V[i][0] = 1
        # PASO 1: inicializar el conjunto de pesos en valores pequeños al azar                                                                
        self.W = np.random.rand(self.M+1, self.nodes_per_layer, self.nodes_per_layer)-0.5   # [capa destino, dest, origen]
        w = np.random.rand(self.nodes_per_layer, len(train_data[0]) - 1)-0.5                      # [dest, origen]
        self.W[1,:,:] = np.zeros((self.nodes_per_layer, self.nodes_per_layer))
        
        # inicializo en 0 los errores en las capas 
        self.d = np.zeros((self.M+1, self.nodes_per_layer))
        # creo las conexiones de los nodos de entrada y la capa oculta
        for nodo_origen in range(len(train_data[0]) - 1):
            for nodo_destino in range(self.nodes_per_layer):
                # capa destino, nodo destino, nodo origen
                self.W[1,nodo_destino,nodo_origen] = w[nodo_destino, nodo_origen]
        
        error_min = 1000000
        delta_accuracy_error = self.delta_accuracy_error 
        total_error = 1
        train_error_per_epoch = []
        test_error_per_epoch = []
        train_worst_error_per_epoch = []
        test_worst_error_per_epoch = []
        train_accuracy = []
        test_accuracy = []
        plotter = Plotter()
        
        for epoch in range(1, self.epochs):
            total_error = 0
            train_worst_error_this_epoch = 0
            corrects = 0
            incorrects = 0
            # mezclo el conjunto de entrenamiento al azar para seleccionar el primero
            np.random.shuffle(train_data)
            # tomo un ejemplo (u)
            for u in range(len(train_data)):
                # Paso 2 (V0 tiene los ejemplos iniciales)
                for k in range(len(train_data[0])-1):
                    # nodos de entrada
                    self.V[0][k] = train_data[u][k]
                # PASO 3: propagar la entrada hasta la capa de salida
                self.propagate()
                # en la última capa hay distinta cantidad de nodos 
                for i in range(0, self.exit_nodes):
                    hMi = self.h(self.M, i, self.nodes_per_layer, self.W, self.V)
                    self.V[self.M][i] = self.g(hMi)
                if self.V[self.M][i] >= train_data[u][-1] - delta_accuracy_error and self.V[self.M][i] <= train_data[u][-1] + delta_accuracy_error:
                    corrects += 1
                else:
                    incorrects += 1

                # PASO 4: calcular el error para la capa de salida
                for i in range(0, self.exit_nodes):
                    #  n[-1] es el último item de la lista n --> data[u][-1] es la salida deseada de mi ejemplo tomado
                    hMi = self.h(self.M, i, self.nodes_per_layer, self.W, self.V)
                    self.d[self.M][i] = self.g_derivative(hMi)*(train_data[u][-1] - self.V[self.M][i])
                # PASO 5: retropropagar el error 
                error_sum = self.back_propagate()
                # PASO 6: actualizar los pesos de las conexiones
                self.update_weights()
                # PASO 7: Calcular el error. Si error > COTA, ir al paso 2
                for i in range(0, self.exit_nodes):
                    if abs(train_data[u][-1] - self.V[self.M][i]) > train_worst_error_this_epoch:
                        train_worst_error_this_epoch = abs(train_data[u][-1] - self.V[self.M][i])
                    total_error += abs(train_data[u][-1] - self.V[self.M][i])
            train_error_per_epoch.append(total_error/len(train_data))
            train_worst_error_per_epoch.append(train_worst_error_this_epoch)
            train_accuracy.append(corrects / (0.0 + corrects + incorrects))
            if self.adaptive and epoch % 10 == 0:
                self.adjust_learning_rate(train_error_per_epoch)
            if total_error < error_min:
                error_min = total_error
                self.w_min = self.W
            if total_error/len(train_data) <= self.error_tolerance or epoch == self.epochs-1:
                self.test_perceptron(test_data, self.w_min, epoch, test_error_per_epoch, test_worst_error_per_epoch, test_accuracy, delta_accuracy_error, True, train_accuracy, train_error_per_epoch)
                break
            else:
                self.test_perceptron(test_data, self.w_min, epoch, test_error_per_epoch, test_worst_error_per_epoch, test_accuracy, delta_accuracy_error, False, train_accuracy, train_error_per_epoch)
        plotter.ej3_errors(train_error_per_epoch, test_error_per_epoch)
        plotter.ej3_accuracy(train_accuracy, test_accuracy)
        return

    