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