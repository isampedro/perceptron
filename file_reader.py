import numpy as np
import string
import random

class Reader:
 
    def __init__(self, excercise):
        self.excercise = excercise
        self.readContent = []
    
    def readFile(self, k):
        
        if self.excercise == 'Ej2':
            return self.readExcerciseTwo2(k)
        

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

    def readExcerciseTwo2(self, k):
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

   
        
    def normalize_output(self, outputs): # tengo que normalizar sino esta fuera del rango de activacion de la tanh o sigmoidea
        max_output = np.max(outputs)
        min_output = np.min(outputs)
        for i in range(0, len(outputs)):
            outputs[i] =  (outputs[i] - min_output) / (max_output - min_output)
        return outputs, max_output, min_output