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
	C, H, W, K = BMeasurements.shape
	BMeasurements_reshaped = torch.reshape(BMeasurements, (-1, K))
	B, K = BMeasurements_reshaped.shape
	N = NormCorrFs.shape[0]
	## Normalize Brightness Measurements functions
	# NormBMeasurements = (BMeasurements.transpose() - np.mean(BMeasurements, axis=1)) / np.std(BMeasurements, axis=1)
	NormBMeasurements_reshaped = (BMeasurements_reshaped.t() - torch.mean(BMeasurements_reshaped, dim=1)) / torch.std(BMeasurements_reshaped, dim=1)
	## Calculate the cross correlation for every measurement and the maximum one will be the depth
	# decodedDepths = torch.zeros((NormBMeasurements.shape[1],), dtype=torch.float32)
	decodedDepths_reshaped = torch.zeros((B,), dtype=torch.float32)
	enum = torch.linspace(0, N - 1, steps=N)
	for i in range(B):
		see = torch.mv(NormCorrFs, NormBMeasurements_reshaped[:,i])
		SM = torch.nn.Softmax()
		temp = SM(see)
		decodedDepths_reshaped[i] = torch.dot(temp, enum)
	decodedDepths = torch.reshape(decodedDepths_reshaped, (C, H, W))
	return decodedDepths
