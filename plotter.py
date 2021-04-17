import matplotlib.pyplot as plt
import numpy as np

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

def plotEx2( ans ):
    x = np.linspace(-2, 2, 200)
    y = (-x*ans[1]-ans[0])/ans[2]
    plt.xlabel('E1')
    plt.ylabel('E2')
    plt.plot(x, y)

    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
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