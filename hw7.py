import numpy as np
from data.sep3way import X,Y,Z
import matplotlib.pyplot as plt

M = 20
N = 20
P = 20

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(X.T[:,0],X.T[:,1],color='r')
ax.scatter(Y.T[:,0],Y.T[:,1],color='g')
ax.scatter(Z.T[:,0],Z.T[:,1],color='b')
plt.show()
