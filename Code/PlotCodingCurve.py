import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

data = np.load('coding_functions.npz')
CorrFs = data['CorrFs']

fig = plt.figure()
ax = fig.gca(projection='3d')
x = CorrFs[:,0]
y = CorrFs[:,1]
z = CorrFs[:,2]
ax.plot(x, y, z)
plt.show()

