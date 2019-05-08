import numpy as np

A = np.zeros((100, 100))
A[0:25,:] = 1000
A[25:50,:] = 1100
A[50:75,:] = 1200
A[75:100,:] = 1300

A = np.ndarray.flatten(A)
np.save('sample_pic.npy', A)

