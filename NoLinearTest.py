import numpy as np
import random
from file_reader import Reader
from plotter import Plotter


class SimplePerceptronNoLinear:

    def __init__(self, eta, epochs, beta, adaptive, k, linear, cross):
        self.eta = eta
        self.initial_eta = eta
        self.epochs = epochs
        self.beta = beta
        self.adaptive = adaptive
        self.k = k
        self.linear = linear
        self.cross = cross



    def sigmoid(self, x):
        return 1 / (1 + np.exp(-2 * self.beta * x))

    def sigmoid_derivated(self, x):
        return 2 * self.beta * self.sigmoid(x) * (1 - self.sigmoid(x))

    def getSum(self, xi, weights):
        sumatoria = 0.0
        # zip me junta dos tuplas
        for i,w in zip(xi, weights):
            sumatoria += i * w
        return sumatoria
    
    def getActivation(self, sumatoria):
        return self.sigmoid(sumatoria)[0]

    def denormalize(self, x, max_, min_):
        return x * (max_ - min_) + min_
    
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
            
    def crossValidation(self, operand):
        r = Reader('Ej2')
        train_data, test_data, max_, min_ = r.readFile(0, self.k, self.linear, self.cross) # agarramos los datos de los txt
        blocks = np.split(np.array(train_data.copy()), self.k)
        plotter = Plotter()
        init_weights = np.random.rand(len(train_data[0]) -1, 1)
        weights = init_weights.copy()
        error_min = 100000000
        error_this_epoch = 1
        w_min = init_weights.copy()
        error_per_epoch = []
        eta_per_epoch = []
        test_error_per_epoch = []
        selectedBlock = 0
        
        for selectedBlock in range(self.k):
            test_data = blocks[selectedBlock].copy()
            train_data = []
            for i in range(self.k):
                    if i != selectedBlock:
                        train_data.extend(blocks[i].copy())
            for epoch in range(self.epochs):
                np.random.shuffle(train_data)
                np.random.shuffle(test_data)
                if error_this_epoch > 0:
                    total_error = 0
                    for i in range(len(train_data)):
                        sumatoria = np.dot(train_data[i][:-1], weights)
                        activation = self.getActivation(sumatoria)
                        error = train_data[i][-1] - activation
                        fixed_diff = self.eta * error
                        derivated = self.sigmoid_derivated(sumatoria)
                        const = np.dot(fixed_diff, derivated)
                        for j in range(len(weights)):
                            weights[j] += (const * train_data[i][j])
                        total_error += error ** 2
                    error_this_epoch = total_error / len(train_data)
                    if epoch > 1:
                        error_per_epoch.append(error_this_epoch)
                        #if self.adaptive and epoch % 10 == 0:
                        #   self.adjust_learning_rate(error_per_epoch_linear)
                    eta_per_epoch.append(self.eta)
                    if epoch == 0:
                        error_min = error_this_epoch
                    if error_this_epoch < error_min:
                        error_min = error_this_epoch
                        w_min = weights
                    if epoch > 1:
                        test_error_per_epoch.append(self.test_perceptron(test_data, w_min, max_, min_, print_=False))
            print('*************** RESULTS ***************')
            print('Analysis for training set:')
            print('Epochs: {}'.format(epoch + 1))
            print('Adaptive: {}'.format(self.adaptive))
            print('Cross: {}'.format(self.cross))
            print('Initial alpha linear: {}'.format(self.initial_eta))
            print('End alpha linear: {}'.format(self.eta))
            print('***************************************')
            self.test_perceptron(test_data, w_min, max_, min_, print_= False)
            print('***************************************')
            plotter.create_plot_ej2(error_per_epoch, test_error_per_epoch, eta_per_epoch, linear=False)
            #plotter.create_plot_ej2(error_per_epoch_non_linear, test_error_per_epoch_non_linear, alpha_per_epoch_non_linear, linear=False)
            weights = init_weights.copy()
            error_min = 100000000
            error_this_epoch = 1
            w_min = init_weights.copy()
            error_per_epoch = []
            eta_per_epoch = []
            test_error_per_epoch = []
        return

    def algorithm(self, operand):
        r = Reader('Ej2')
        train_data, test_data, max_, min_ = r.readFile(0, self.k, self.linear, self.cross) # agarramos los datos de los txt
        plotter = Plotter()
        init_weights = np.random.rand(len(train_data[0]) -1, 1)
        weights = init_weights.copy()
        error_min = 1000000
        error_this_epoch = 1
        w_min = init_weights.copy()
        error_per_epoch = []
        eta_per_epoch = []
        test_error_per_epoch = []

        for epoch in range(self.epochs):
            np.random.shuffle(train_data)
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
                    derivated = self.sigmoid_derivated(sumatoria)
                    const = np.dot(fixed_diff, derivated)
                    # w = w + ∆w --> actualizo los pesos
                    for j in range(len(weights)):
                        weights[j] += (const * train_data[i][j])
                    total_error += self.denormalize(error, max_, min_)**2
                
                error_this_epoch = total_error / len(train_data)
                error_per_epoch.append(error_this_epoch)
                #if self.adaptive and epoch % 10 == 0:
                 #   self.adjust_learning_rate(error_per_epoch_linear)
                eta_per_epoch.append(self.eta)
                if epoch == 0:
                    error_min = error_this_epoch
                if error_this_epoch < error_min:
                    error_min = error_this_epoch
                    w_min = weights
                test_error_per_epoch.append(self.test_perceptron(test_data, w_min, max_, min_, print_=False))

        print('*************** RESULTS ***************')
        print('Analysis for training set:')
        print('Epochs: {}'.format(epoch + 1))
        print('Adaptive: {}'.format(self.adaptive))
        print('Cross: {}'.format(self.cross))
        print('Initial alpha linear: {}'.format(self.initial_eta))
        print('End alpha linear: {}'.format(self.eta))
        print('***************************************')
        self.test_perceptron(test_data, w_min, max_, min_, print_= False)
        print('***************************************')
        
        plotter.create_plot_ej2(error_per_epoch, test_error_per_epoch, eta_per_epoch, linear=False)
        #plotter.create_plot_ej2(error_per_epoch_non_linear, test_error_per_epoch_non_linear, alpha_per_epoch_non_linear, linear=False)
        return

    def test_perceptron(self, test_data, weights, max_out, min_out, print_):
        if print_: print('Testing perceptron {}...'.format(('no linear')))
        error = 0.0
        error_accum = 0.0
        if print_:
            print('+-------------------+-------------------+')
            print('|   Desired output  |   Perceptron out  |')
            print('+-------------------+-------------------+')
        for row in test_data:
            sumatoria = self.getSum(row[:-1], weights)
            perceptron_output = self.getActivation(sumatoria)
            denorm_real = self.denormalize(row[-1], max_out, min_out)
            denom_perc = self.denormalize(perceptron_output, max_out, min_out)
            if print_: print('|{:19f}|{:19f}|'.format(denorm_real, denom_perc))
            diff = denorm_real - denom_perc
            error_accum += abs(diff)
            error += (diff)**2
        if print_: 
            print('+-------------------+-------------------+')
            print('Test finished')
            print('Error avg {}: {}'.format('linear', error_accum/len(test_data)))
        return error/len(test_data)
