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
            #if linear: alphas[i] *= 50000
            #else: alphas[i] *= 10000
            epochs.append(i)
        plt.plot(epochs, errors, label='Errores de entrenamiento')
        plt.plot(epochs, test_errors, label='Errores de testeo')
        #plt.plot(epochs, alphas, label='Variacion del aprendizaje')
        plt.legend(fontsize = 8, loc = 0)
        plt.grid(True)
        plt.show()

    