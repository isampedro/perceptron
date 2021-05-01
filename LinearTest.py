import numpy as np
import random
from file_reader import Reader
from plotter import Plotter


class SimplePerceptron:

    def __init__(self, eta, epochs, beta, adaptive, k):
        self.eta = eta
        self.initial_eta = eta
        self.epochs = epochs
        self.beta = beta
        self.adaptive = adaptive
        self.k = k



    def identity(self, x):
        return x

    def getSum(self, xi, weights):
        sumatoria = 0.0
        # zip me junta dos tuplas
        for i,w in zip(xi, weights):
            sumatoria += i * w
        return sumatoria
    
    def getActivation(self, sumatoria):
        return self.identity(sumatoria)[0]
    
    def error_function(self, sqr_errors_sum):
        if isinstance(sqr_errors_sum, list):
            return (0.5 * (sqr_errors_sum))[0] 
        else:
            return 0.5 * (sqr_errors_sum)

    def adjust_learning_rate(self, errors_so_far):
        if(len(errors_so_far) > 10):
            last_10_errors = errors_so_far[-10:]
            booleans = []
            for i in range(len(last_10_errors) - 1):
                booleans.append(last_10_errors[i] > last_10_errors[i + 1])
            if all(booleans):
                self.eta += 0.001
            elif not all(booleans):
                self.eta -= 0.01 * self.eta
            


    def algorithm(self, operand):
        r = Reader('Ej2')
        train_data, test_data = r.readFile(self.k) # agarramos los datos de los txt
        plotter = Plotter()
        init_weights = np.random.rand(len(train_data[0]) -1, 1)
        weights = init_weights.copy()
        error_min = 100000000
        error_this_epoch = 1
        w_min = init_weights.copy()
        error_per_epoch = []
        eta_per_epoch = []
        test_error_per_epoch = []

        for epoch in range(self.epochs):
            if error_this_epoch > 0:
                total_error = 0
                for i in range(len(train_data)):
                    # exitacion = x(i x,:) * w
                    sumatoria = np.dot(train_data[i][:-1], weights)
                    # activacion = signo(exitacion);
                    activation = self.getActivation(sumatoria)
                    # expected_value - my_output_value
                    error = train_data[i][-1] - activation
                    # ∆w = η * (y(1,i x) - activacion) * x(i x,:)’;     donde (y(1,i x) - activacion) = error_linear
                    fixed_diff = self.eta * error
                    # w = w + ∆w --> actualizo los pesos
                    for j in range(len(weights)):
                        weights[j] += (fixed_diff * train_data[i][j])
                    total_error += error ** 2
                
                error_this_epoch = total_error / len(train_data)
                if epoch > 1:
                    error_per_epoch.append(error_this_epoch)
                print(error_this_epoch)
                #if self.adaptive and epoch % 10 == 0:
                 #   self.adjust_learning_rate(error_per_epoch_linear)
                eta_per_epoch.append(self.eta)
                if epoch == 0:
                    error_min = error_this_epoch
                if error_this_epoch < error_min:
                    error_min = error_this_epoch
                    w_min = weights
                if epoch > 1:
                    test_error_per_epoch.append(self.test_perceptron(test_data, w_min, print_=False))

        print('*************** RESULTS ***************')
        print('Analysis for training set:')
        print('Epochs: {}'.format(epoch + 1))
        print('Initial alpha linear: {}'.format(self.initial_eta))
        print('End alpha linear: {}'.format(self.eta))
        print('***************************************')
        self.test_perceptron(test_data, w_min, print_= False)
        print('***************************************')
        
        plotter.create_plot_ej2(error_per_epoch, test_error_per_epoch, eta_per_epoch, linear=True)
        #plotter.create_plot_ej2(error_per_epoch_non_linear, test_error_per_epoch_non_linear, alpha_per_epoch_non_linear, linear=False)
        return

    def test_perceptron(self, test_data, weights, print_):
        if print_: print('Testing perceptron {}...'.format(('linear')))
        error = 0.0
        error_accum = 0.0
        if print_:
            print('+-------------------+-------------------+')
            print('|   Desired output  |   Perceptron out  |')
            print('+-------------------+-------------------+')
        for row in test_data:
            sumatoria = self.getSum(row[:-1], weights)
            perceptron_output = self.getActivation(sumatoria)
            if print_: print('|{:19f}|{:19f}|'.format(row[-1], perceptron_output))
            diff = row[-1] - perceptron_output
            error_accum += abs(diff)
            error += (diff)**2
        if print_: 
            print('+-------------------+-------------------+')
            print('Test finished')
            print('Error avg {}: {}'.format('linear', error_accum/len(test_data)))
        return error/len(test_data)

    
