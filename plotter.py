from matplotlib import pyplot as plt
import matplotlib
import numpy as np

class Plotter:

    
    
    def create_plot_ej2(self, errors, test_errors, alphas, linear):
        fig,ax = plt.subplots()
        ax.set_title('Evolucion de error por epoca - {}'.format('LINEAL' if linear else 'NO LINEAL'))
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Error")
        epochs = []
        for i in range(len(errors)):
            if linear: alphas[i] *= 50000
            else: alphas[i] *= 10000
            epochs.append(i)
        plt.plot(epochs, errors, label='Errores de entrenamiento')
       # plt.plot(epochs, test_errors, label='Errores de testeo')
        #plt.plot(epochs, alphas, label='Variacion del aprendizaje')
        plt.legend(fontsize = 8, loc = 0)
        plt.grid(True)
        plt.show()

    def ej3_errors(self, errors, test_errors):
        fig,ax = plt.subplots()
        ax.set_title('Evolucion del error por épocas')
        ax.set_xlabel("Época")
        ax.set_ylabel("Error")
        epochs = []
        for i in range(len(errors)):
            epochs.append(i)
        plt.plot(epochs, errors, 'cornflowerblue', label='Error de entrenamiento')
        plt.plot(epochs, test_errors, 'darkorange', label='Error de testeo')
        plt.legend(fontsize = 8)
        plt.grid(True)
        plt.show()

    def ej3_accuracy(self, accuracy, test_accuracy):
        fig,ax = plt.subplots()
        ax.set_title('Evolución de accuracy por épocas')
        ax.set_xlabel("Época")
        ax.set_ylabel("Accuracy")
        epochs = []
        for i in range(len(accuracy)):
            epochs.append(i)
        plt.plot(epochs, accuracy, label='Accuracy entrenamiento')
        plt.plot(epochs, test_accuracy, label='Accuracy testeo')
        plt.legend(fontsize = 8, loc = 0)
        plt.grid(True)
        plt.show()

    