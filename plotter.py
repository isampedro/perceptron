import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plotEx1( ans, exitValues ):
    x = np.linspace(-2, 2, 200)
    y = (-x*ans[1]-ans[0])/ans[2]
    plt.xlabel('E1')
    plt.ylabel('E2')
    plt.plot(x, y)

    if exitValues[0][0] == 1:
        plt.plot(-1,1,marker='o', color='yellow')
    else:
        plt.plot(-1,1,marker='o', color='red')
    plt.annotate("(-1,1)", (-1,1))
    if exitValues[2][0] == 1:
        plt.plot(-1,-1,marker='o', color='yellow')
    else:
        plt.plot(-1,-1,marker='o', color='red')
    plt.annotate("(-1,-1)", (-1,-1))
    if exitValues[3][0] == 1:
        plt.plot(1,1,marker='o', color='yellow')
    else:
        plt.plot(1,1,marker='o', color='red')
    plt.annotate("(1,1)", (1,1))
    if exitValues[1][0] == 1:
        plt.plot(1,-1,marker='o', color='yellow')
    else:
        plt.plot(1,-1,marker='o', color='red')
    plt.annotate("(1,-1)", (1,-1))


    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.show()

def plotEx2( ans, inputPoints ):
    fig = plt.figure()
    ax = Axes3D(fig)

    x = np.linspace(-10, 10, 2000)
    y = np.linspace(-10, 10, 2000)
    z = (-x*ans[1]-ans[0] - y*ans[2])/ans[3]
    ax.plot(x, y, z)
    for point in inputPoints:
        ax.plot(point[0], point[1], point[2], marker='o', color='red')

    plt.show()

def plotEx3Errors(errors, worst_errors, testEPE, testWorstEPE):
        fig,ax = plt.subplots()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        epochs = []
        for i in range(len(errors)):
            epochs.append(i)
        plt.plot(epochs, errors, label='Avg Error Training')
        plt.plot(epochs, worst_errors, label='Maxi Error Training')
        plt.plot(epochs, testWorstEPE, label='Max Error Testing')
        plt.plot(epochs, testEPE, label='AVg Error Testing')
        plt.legend(fontsize = 10)
        plt.grid(True)
        plt.show()

def plotAccuracy(accuracyTraining, accuracyTesting):
        fig,ax = plt.subplots()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Precision")
        epochs = []
        for i in range(len(accuracyTesting)):
            epochs.append(i)
        plt.plot(epochs, accuracyTesting, label="Accuracy Testing")
        plt.plot(epochs, accuracyTraining, label="Accuracy Training")
        plt.legend(fontsize = 10)
        plt.grid(True)
        plt.show()


def plotErrors( errors ):
    x = []
    y = []
    for error in errors:
        x.append(error['iteration'])
        y.append(error['error'])
     
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.plot(x, y)

    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.show()

