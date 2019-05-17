import numpy as np

A = np.zeros((100, 100))
A[0:25,:] = 2000
A[25:50,:] = 2100
A[50:75,:] = 2200
A[75:100,:] = 2300

A = np.ndarray.flatten(A)
np.save('sample_pic.npy', A)

