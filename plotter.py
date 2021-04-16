import matplotlib.pyplot as plt
import numpy as np

def plot( ans ):
    x = np.linspace(-2, 2, 200)
    y = (-x*ans[1]-ans[0])/ans[2]
    plt.xlabel('E1')
    plt.ylabel('E2')
    plt.plot(x, y)
    plt.plot(-1,1,marker='o')
    plt.annotate("(-1,1)", (-1,1))
    plt.plot(-1,-1,marker='o')
    plt.annotate("(-1,-1)", (-1,-1))
    plt.plot(1,1,marker='o')
    plt.annotate("(1,1)", (1,1))
    plt.plot(1,-1,marker='o')
    plt.annotate("(1,-1)", (1,-1))
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.show()