import matplotlib
import matplotlib.pyplot as plt
import torch

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

# Generate square wave
x1 = -1*torch.ones((1,100), dtype=torch.float)
x2 = 1*torch.ones((1,100), dtype=torch.float)
x = torch.cat((x1,x2,x1,x2,x1,x2,x1,x2,x1,x2,x1,x2,x1,x2,x1,x2),1)

# Calculate FFT
y = torch.rfft(x[0,:], 1)
y_real,y_imag = y.t()

# Shape checks
print(y.shape)
print(y_real.shape)
print(y_imag.shape)

# Cross correlation test (autocorrelation of x)
temp = complex_conj_multiplication(torch.rfft(x[0,:],1), torch.rfft(x[0,:],1))
CorrFs = torch.irfft(temp,1)
#### Get correlation function


# Plot
x = x.numpy()
y = y.numpy()
CorrFs = CorrFs.numpy()
fig, ax = plt.subplots(4,1)
ax[0].plot(x[0,:])
ax[1].plot(y[:,1])
ax[2].plot(y[:,0])
ax[3].plot(CorrFs)
plt.show()
