import perceptron as p
import calculateError
import simplePerceptronTypes
import numpy as np
import matplotlib.pyplot as plt

x = [100, -1, 1], [100, 1, -1], [100, 1, 1], [100, -1, -1]
y = [-1, -1, 1, -1]
plt.xlabel('E1')
plt.ylabel('E2')
ans = p.simplePerceptron(x,y, 2, 4, 1, calculateError.cuadraticError, simplePerceptronTypes.scalar)

x = np.linspace(-1, 1, 100)
y = -x*ans[1]-ans[0]

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
