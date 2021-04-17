import matplotlib.pyplot as plt
import numpy as np

def plot( ans, exitValues ):
    x = np.linspace(-2, 2, 200)
    y = (-x*ans[1]-ans[0])/ans[2]
    plt.xlabel('E1')
    plt.ylabel('E2')
    plt.plot(x, y)

    if exitValues[0] == 1:
        plt.plot(-1,1,marker='o', color='yellow')
    else:
        plt.plot(-1,1,marker='o', color='red')
    plt.annotate("(-1,1)", (-1,1))
    if exitValues[2] == 1:
        plt.plot(-1,-1,marker='o', color='yellow')
    else:
        plt.plot(-1,-1,marker='o', color='red')
    plt.annotate("(-1,-1)", (-1,-1))
    if exitValues[3] == 1:
        plt.plot(1,1,marker='o', color='yellow')
    else:
        plt.plot(1,1,marker='o', color='red')
    plt.annotate("(1,1)", (1,1))
    if exitValues[1] == 1:
        plt.plot(1,-1,marker='o', color='yellow')
    else:
        plt.plot(1,-1,marker='o', color='red')
    plt.annotate("(1,-1)", (1,-1))


    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.show()