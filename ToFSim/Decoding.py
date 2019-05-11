"""Decoding
Decoding functions for time of flight coding schemes.
"""
#### Python imports

#### Library imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

#### Local imports

def DecodeXCorr(BMeasurements, NormCorrFs):
	"""DecodeXCorr: Generic decoding algorithm that performs a 1D search on the normalized 
	correlation functions.
	
	Args:
	    BMeasurements (np.ndarray): B x K matrix. B sets of K brightness measurements
	    NormCorrFs (np.ndarray): N x K matrix. Normalized Correlation functions. Zero mean
	    unit variance.
	Returns:
	    np.array: decodedDepths 
	"""
	(N,K) = NormCorrFs.shape
	## Normalize Brightness Measurements functions
	NormBMeasurements = (BMeasurements.transpose() - np.mean(BMeasurements, axis=1)) / np.std(BMeasurements, axis=1)
	## Calculate the cross correlation for every measurement and the maximum one will be the depth
	decodedDepths = np.zeros((NormBMeasurements.shape[1],))
	for i in range(NormBMeasurements.shape[1]):
		#decodedDepths[i] = np.argmax(np.dot(NormCorrFs, NormBMeasurements[:,i]), axis=0)
		Corr_B = np.dot(NormCorrFs, NormBMeasurements[:,i])
		softmax = scipy.special.softmax(100*Corr_B,axis=0)
		decodedDepths[i] = np.dot(softmax,np.arange(N))

		#fig, ax = plt.subplots()
		#ax.plot(softmax)
		#ax.grid()
		#plt.show()
		print(decodedDepths[i])

		#print(min(Corr_B))
		#print(max(Corr_B))
		#fig, ax = plt.subplots()
		#ax.plot(softmax)
		#ax.grid()
		#plt.show()
		#print(softmax.shape)
		#print(np.max(softmax))

		

	return decodedDepths
