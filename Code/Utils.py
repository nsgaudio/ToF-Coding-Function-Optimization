#### Python imports
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#### Library imports
import numpy as np
import scipy as sp
from scipy import stats
from scipy import fftpack 
from scipy import signal
from scipy import linalg
from scipy import interpolate
import torch

def ScaleAreaUnderCurve(x, dx=0., desiredArea=1.):
	"""ScaleAreaUnderCurve: Scale the area under the curve x to some desired area.
	
	Args:
	    x (TYPE): Discrete set of points that lie on the curve. Numpy vector
	    dx (float): delta x. Set to 1/length of x by default.
	    desiredArea (float): Desired area under the curve.
	
	Returns:
	    numpy.ndarray: Scaled vector x with new area.
	"""
	#### Validate Input
	# assert(UtilsTesting.IsVector(x)),'Input Error - ScaleAreaUnderCurve: x should be a vector.'

	#### Calculate some parameters
	N = x.size
	#### Set default value for dc
	if(dx == 0): dx = 1./float(N)
	#### Calculate new area
	oldArea = torch.sum(x)*dx
	y = x*desiredArea/oldArea
	#### Return scaled vector
	return y 


def ScaleMod(ModFs, device, tau=1., pAveSource=1.):
	"""ScaleMod: Scale modulation appropriately given the beta of the scene point, the average
	source power and the repetition frequency.
	
	Args:
	    ModFs (np.ndarray): N x K matrix. N samples, K modulation functions
	    tau (float): Repetition frequency of ModFs 
	    pAveSource (float): Average power emitted by the source 
	    beta (float): Average reflectivity of scene point

	Returns:
	    np.array: ModFs 
	"""
	(N,K) = ModFs.shape
	dt = tau/float(N)
	eTotal = tau*pAveSource # Total Energy

	ModFs_clone = ModFs#ModFs.clone() # Clone ModFs, otherwise leaf variable error
	ModFs_scaled = torch.zeros([N,K], dtype=torch.float, device=device) # Instantiate scaled ModFs
	for i in range(0,K): 
		ModFs_scaled[:,i] = ScaleAreaUnderCurve(x=ModFs[:,i], dx=dt, desiredArea=eTotal)
		#ModFs[:,i] = ScaleAreaUnderCurve(x=ModFs_clone[:,i], dx=dt, desiredArea=eTotal)

	return ModFs_scaled


def ApplyKPhaseShifts(x, shifts):
	"""ApplyPhaseShifts: Apply phase shift to each vector in x. 
	
	Args:
	    x (np.array): NxK matrix
	    shifts (np.array): Array of dimension K.
	
	Returns:
	    np.array: Return matrix x where each column has been phase shifted according to shifts. 
	"""
	K = 0
	if(type(shifts) == np.ndarray): K = shifts.size
	elif(type(shifts) == list): K = len(shifts) 
	else: K = 1
	for i in range(0,K):
		x[:,i] = np.roll(x[:,i], int(round(shifts[i])))

	return x

def complex_conj_multiplication(t1, t2):
	"""Multiplies the complex conjugate of tensor t1 with tensor t2

	Args:
		t1: Tensor 1 (complex conjugate)
		t2: Tensor 2 (complex)

	Return:
		out: Tensor that contains the multiplication result
	"""
	real1, imag1 = t1.t()
	real2, imag2 = t2.t()
	imag1_conj = -1*imag1
	return torch.stack([real1 * real2 - imag1_conj * imag2, real1 * imag2 + imag1_conj * real2], dim = -1)

def GetCorrelationFunctions(ModFs, DemodFs, device, dt=None):
	"""GetCorrelationFunctions: Calculate the circular correlation of all modF and demodF.
	
	Args:
		ModFs: Array of modulation functions. NxK matrix. NO DIMENSION CHECK performed.
		DemodFs: Array of demodulation functions. NxK matrix. NO DIMENSION CHECK performed.
	
	Returns:
	    Tensor: NxK matrix. Each column is the correlation function for the respective pair.
	"""

	#### Declare some parameters
	(N,K) = ModFs.shape
	#### Get dt
	if(dt == None): dt = 1./N
	#### Allocate the correlation function matrix
	CorrFs = torch.zeros(ModFs.shape,device=device)
	#### Get correlation functions
	for i in range(0,K):
		temp = complex_conj_multiplication(torch.rfft(ModFs[:,i],1), torch.rfft(DemodFs[:,i],1))
		CorrFs[:,i] = torch.irfft(temp,1)[0:N]
	#### Scale by dt
	CorrFs_scaled = CorrFs*dt
	return CorrFs_scaled


def NormalizeBrightnessVals(BVals):
	## Normalized correlation functions, zero mean, unit variance. We have to transpose so that broadcasting works.
	NormBVals = (BVals.transpose() - np.mean(BVals, axis=1)) / np.std(BVals, axis=1) 
	# Transpose it again so that it has dims NxK
	NormBVals = NormBVals.transpose()
	return NormBVals


def ComputeBrightnessVals(ModFs, DemodFs, CorrFs, depths=None, pAmbient=0, beta=1, T=1, tau=1, dt=1, gamma=1):
	"""ComputeBrightnessVals: Computes the brightness values for each possible depth.
	
	Args:
	    ModFs (np.ndarray): N x K matrix. N samples, K modulation functions
	    DemodFs (np.ndarray): N x K matrix. N samples, K demodulation functions
	    tau (float): Repetitiion period of ModFs and DemodFs
	    pAmbient (float): Average power of the ambient illumination component
	    beta (float): Reflectivity to be used
	    T (float): 
	Returns:
	    np.array: ModFs 
	"""
	(N,K) = ModFs.shape
	if(depths is None): depths = np.arange(0, N, 1)
	depths = torch.round(depths).type(torch.long)
	## Calculate correlation functions (integral over 1 period of m(t-phi)*d(t)) for all phi
	#CorrFs = GetCorrelationFunctions(ModFs,DemodFs,dt=dt)
	## Calculate the integral of the demodulation function over 1 period
	kappas = torch.sum(DemodFs,0)*dt
	## Calculate brightness values
	BVals = (gamma*beta)*(T/tau)*(CorrFs + pAmbient*kappas)

	#fig, ax = plt.subplots()
	#ax.plot(BVals.detach().numpy())
	#ax.grid()
	#plt.show(block=True)

	## Return only the brightness vals for the specified depths
	BVals = BVals[depths,:]
	return (BVals)

def GetClippedBSamples(nSamples, BMean, BVar, device):
	"""GetClippedBSamples: Draw N brightness samples from the truncated multivariate gaussian dist 
	with mean BVal and Covariance Sigma=diag(NoiseVar)
	Args:
	    nSamples (int): Number of samples to draw.
	    BMean (np.ndarray): 1 x K array. 
	    BVar (np.ndarray): 1 x K array. 
	Returns:
	    BSampels (np.ndarray): nSamples x K array.  
	"""
	K = BMean.size
	lower, upper = 0, 1

	noise = torch.randn(BVar.size(),device=device)*torch.sqrt(BVar) 
	BSamples = BMean + noise
	BSamples[BSamples<0]=lower
	BSamples[BSamples>1]=upper

	return (BSamples)
