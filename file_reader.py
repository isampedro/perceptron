import numpy as np
import string
import random

class Reader:
 
    def __init__(self, excercise):
        self.excercise = excercise
        self.readContent = []
    
    def readFile(self, size, k, linear, cross):
        if self.excercise == 'Ej2':
            if linear:
                if cross:
                    return self.readExcerciseTwoCross(k)
                else:
                    return self.readExcerciseTwo(k)
            else:
                if cross:
                    return self.readExcerciseTwoNonLinearCross(k)
                else:
                    return self.readExcerciseTwoNonLinear(k)
        if self.excercise == 'Ej3':
            return self.readExerciseThree(size, cross, k)
        
    def readExcerciseTwoCross(self, k):
        f = open('TP3-ej2-Conjunto-entrenamiento.txt', 'r')
        g = open('TP3-ej2-Salida-deseada.txt', 'r')
        # separo por filas
        linesf = f.read().split('\n')
        linesg = g.read().split('\n')
        total_amount = len(linesf)
        testing_amount = int (total_amount / k)
        training_amount = total_amount - testing_amount
        
        # strip() lo uso para quitar el espacio inicial que me genero split en cada uno de los arreglos
        # tengo 200 arreglos donde cada uno tiene 3 elementos
        valuesf = [line.strip() for line in linesf]
        valuesg = [float(line.strip()) for line in linesg]
        # separo cada arreglo y lo hago un elemento individual
        values_indiv = [line.split(' ') for line in valuesf]
        index = 0
        train = []
        training_indexes = set()
        while len(training_indexes) < training_amount:
            # tomo indices al azar para la data de mi conjunto de entrenamiento
            training_indexes.add(random.randint(0, total_amount - 1))
        training_indexes_list = list(training_indexes)
        for i in range(training_amount):
            # agrego el bias
            train.append([1.0])
            # agrego la información 
            row = values_indiv[training_indexes_list[i]]
            for element in row:
                if element != '':
                    train[i].append(float(element))
            # agrego el valor esperado
            train[i].append(float(valuesg[training_indexes_list[i]]))
        
        # Hago lo mismo ahora pero para el conjunto de prueba
        testing_indexes_list = list(set(range(0, total_amount)) - training_indexes)
        test = []
        for i in range(testing_amount):
            # agrego el bias
            test.append([1.0])
            # agrego la información 
            row = values_indiv[testing_indexes_list[i]]
            for element in row:
                if element != '':
                    test[i].append(float(element))
            # agrego el valor esperado
            test[i].append(float(valuesg[testing_indexes_list[i]]))
        train.extend(test)
        return train, test

    def readExcerciseTwo(self, k):
        f = open('TP3-ej2-Conjunto-entrenamiento.txt', 'r')
        g = open('TP3-ej2-Salida-deseada.txt', 'r')
        # separo por filas
        linesf = f.read().split('\n')
        linesg = g.read().split('\n')
        total_amount = len(linesf)
        testing_amount = int (total_amount / k)
        training_amount = total_amount - testing_amount
        
        # strip() lo uso para quitar el espacio inicial que me genero split en cada uno de los arreglos
        # tengo 200 arreglos donde cada uno tiene 3 elementos
        valuesf = [line.strip() for line in linesf]
        valuesg = [float(line.strip()) for line in linesg]
        # separo cada arreglo y lo hago un elemento individual
        values_indiv = [line.split(' ') for line in valuesf]
        index = 0
        train = []
        training_indexes = set()
        while len(training_indexes) < training_amount:
            # tomo indices al azar para la data de mi conjunto de entrenamiento
            training_indexes.add(random.randint(0, total_amount - 1))
        training_indexes_list = list(training_indexes)
        for i in range(training_amount):
            # agrego el bias
            train.append([1.0])
            # agrego la información 
            row = values_indiv[training_indexes_list[i]]
            for element in row:
                if element != '':
                    train[i].append(float(element))
            # agrego el valor esperado
            train[i].append(float(valuesg[training_indexes_list[i]]))
        
        # Hago lo mismo ahora pero para el conjunto de prueba
        testing_indexes_list = list(set(range(0, total_amount)) - training_indexes)
        test = []
        for i in range(testing_amount):
            # agrego el bias
            test.append([1.0])
            # agrego la información 
            row = values_indiv[testing_indexes_list[i]]
            for element in row:
                if element != '':
                    test[i].append(float(element))
            # agrego el valor esperado
            test[i].append(float(valuesg[testing_indexes_list[i]]))
            
        return train, test

    def readExcerciseTwoNonLinearCross(self, k):
        f = open('TP3-ej2-Conjunto-entrenamiento.txt', 'r')
        g = open('TP3-ej2-Salida-deseada.txt', 'r')
        # separo por filas
        linesf = f.read().split('\n')
        linesg = g.read().split('\n')
        total_amount = len(linesf)
        testing_amount = int (total_amount / k)
        training_amount = total_amount - testing_amount
        
        # strip() lo uso para quitar el espacio inicial que me genero split en cada uno de los arreglos
        # tengo 200 arreglos donde cada uno tiene 3 elementos
        valuesf = [line.strip() for line in linesf]
        valuesg, max_out, min_out = self.normalize_output ([float(line.strip()) for line in linesg]) 
        # separo cada arreglo y lo hago un elemento individual
        values_indiv = [line.split(' ') for line in valuesf]
        index = 0
        train = []
        training_indexes = set()
        while len(training_indexes) < training_amount:
            # tomo indices al azar para la data de mi conjunto de entrenamiento
            training_indexes.add(random.randint(0, total_amount - 1))
        training_indexes_list = list(training_indexes)
        for i in range(training_amount):
            # agrego el bias
            train.append([1.0])
            # agrego la información 
            row = values_indiv[training_indexes_list[i]]
            for element in row:
                if element != '':
                    train[i].append(float(element))
            # agrego el valor esperado
            train[i].append(float(valuesg[training_indexes_list[i]]))
        
        # Hago lo mismo ahora pero para el conjunto de prueba
        testing_indexes_list = list(set(range(0, total_amount)) - training_indexes)
        test = []
        for i in range(testing_amount):
            # agrego el bias
            test.append([1.0])
            # agrego la información 
            row = values_indiv[testing_indexes_list[i]]
            for element in row:
                if element != '':
                    test[i].append(float(element))
            # agrego el valor esperado
            test[i].append(float(valuesg[testing_indexes_list[i]]))
        train.extend(test)
        return train, test, max_out, min_out

    def readExcerciseTwoNonLinear(self, k):
        f = open('TP3-ej2-Conjunto-entrenamiento.txt', 'r')
        g = open('TP3-ej2-Salida-deseada.txt', 'r')
        # separo por filas
        linesf = f.read().split('\n')
        linesg = g.read().split('\n')
        total_amount = len(linesf)
        testing_amount = int (total_amount / k)
        training_amount = total_amount - testing_amount
        
        # strip() lo uso para quitar el espacio inicial que me genero split en cada uno de los arreglos
        # tengo 200 arreglos donde cada uno tiene 3 elementos
        valuesf = [line.strip() for line in linesf]
        valuesg, max_out, min_out = self.normalize_output ([float(line.strip()) for line in linesg]) 
        # separo cada arreglo y lo hago un elemento individual
        values_indiv = [line.split(' ') for line in valuesf]
        index = 0
        train = []
        training_indexes = set()
        while len(training_indexes) < training_amount:
            # tomo indices al azar para la data de mi conjunto de entrenamiento
            training_indexes.add(random.randint(0, total_amount - 1))
        training_indexes_list = list(training_indexes)
        for i in range(training_amount):
            # agrego el bias
            train.append([1.0])
            # agrego la información 
            row = values_indiv[training_indexes_list[i]]
            for element in row:
                if element != '':
                    train[i].append(float(element))
            # agrego el valor esperado
            train[i].append(float(valuesg[training_indexes_list[i]]))
        
        # Hago lo mismo ahora pero para el conjunto de prueba
        testing_indexes_list = list(set(range(0, total_amount)) - training_indexes)
        test = []
        for i in range(testing_amount):
            # agrego el bias
            test.append([1.0])
            # agrego la información 
            row = values_indiv[testing_indexes_list[i]]
            for element in row:
                if element != '':
                    test[i].append(float(element))
            # agrego el valor esperado
            test[i].append(float(valuesg[testing_indexes_list[i]]))
            
        return train, test, max_out, min_out

    def ex_3(self, amount, cross_validation, k):
        # un número está representados por imágenes de 5 x 7 pixeles
        width = 5
        height = 7
        f = open('mapa-de-pixeles.txt', 'r')
        linesf = f.read().split('\n')
        valuesf = [line.strip() for line in linesf]
        values_indiv = [line.split(' ') for line in valuesf]
        # +2 porque a cada número le tengo que agregar el bias y el expected output
        data = np.zeros((10, width * height + 2))
        # tengo 10 números (del 0 al 9)
        for index in range(10):
            for row in range(height):
                for col in range(width):
                    data[index][1 + col + row * width] = int(values_indiv[index * height + row][col])
        for i in range(len(data)):
            # agrego el bias
            data[i][0] = 1
            # -1 si es par, 1 si es impar
            data[i][-1] = (i % 2 * 2) - 1
        np.random.shuffle(data)
        if cross_validation:
            test_data = data[0:k]
            train_data = data[k:]
        else:
            train_data = data[0:amount]
            test_data = data[amount:]
       
        return train_data, test_data

    def readExerciseThree(self, amount, cross_validation, k):
        width = 5
        height = 7
        f = open('mapa-de-pixeles.txt', 'r')
        linesf = f.read().split('\n')
        valuesf = [line.strip() for line in linesf]
        values_indiv = [line.split(' ') for line in valuesf]
        data = np.zeros((10, width*height + 2))
        train_data = np.zeros((amount, width*height + 2))
        test_data = np.zeros((10-amount, width*height + 2))
        for index in range(10):                    
            for fila in range(height):
                for col in range(width):
                    data[index][1+col+fila*width] = int(values_indiv[index*height+fila][col])
        for i in range(len(data)):
            data[i][0] = 1
            data[i][-1] = (i%2 * 2) - 1
        np.random.shuffle(data)
        
        if cross_validation:
            test_data = data[:]
            train_data = data[:]
            
        else:
            train_data = data[0:amount]
            test_data = data[amount:]
        
        return train_data, test_data

   
        
    def normalize_output(self, outputs): # tengo que normalizar sino esta fuera del rango de activacion de la tanh o sigmoidea
        max_output = np.max(outputs)
        min_output = np.min(outputs)
        for i in range(0, len(outputs)):
            outputs[i] =  (outputs[i] - min_output) / (max_output - min_output)
        return outputs, max_output, min_output