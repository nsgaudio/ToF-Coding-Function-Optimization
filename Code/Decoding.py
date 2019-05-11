"""Decoding
Decoding functions for time of flight coding schemes.
"""
#### Python imports

#### Library imports
import numpy as np
import torch

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

	B, K = BMeasurements.shape
	N = NormCorrFs.shape[0]
	## Normalize Brightness Measurements functions
	# NormBMeasurements = (BMeasurements.transpose() - np.mean(BMeasurements, axis=1)) / np.std(BMeasurements, axis=1)
	NormBMeasurements = (BMeasurements.t() - torch.mean(BMeasurements, axis=1)) / torch.std(BMeasurements, axis=1)
	NormBMeasurements_clone = NormBMeasurements.clone()
	## Calculate the cross correlation for every measurement and the maximum one will be the depth
	# decodedDepths = torch.zeros((NormBMeasurements.shape[1],), dtype=torch.float32)
	decodedDepths = torch.zeros((B,), dtype=torch.float32)
	for i in range(B):
		# decodedDepths[i] = np.argmax(np.dot(NormCorrFs, NormBMeasurements[:,i]), axis=0)
		decodedDepths[i] = torch.Softmax(torch.mm(NormCorrFs, NormBMeasurements_clone[:,i]), axis=0)
	enum = torch.linspace(0, N - 1, steps=N)
	result = torch.dot(decodedDepths, enum)
	return result
