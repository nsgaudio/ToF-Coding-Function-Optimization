import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Load groundtruth picture and decoded picture
depths = np.load('sample_pic.npy')
depths = np.reshape(depths, (100,-1))
decoded_depths = np.load('sample_pic_decoded.npy')
decoded_depths = np.reshape(decoded_depths, (100,-1))

error = np.abs(depths - decoded_depths)
mse = (np.square(depths - decoded_depths)).mean(axis=None)
print("Pixelwise mean squared error: %f" %(mse))

# Plot Groundtruth
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(100)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, depths)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim(950,1350)
plt.show()

# Plot Decoded
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(100)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, decoded_depths)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim(950,1350)
plt.show()
