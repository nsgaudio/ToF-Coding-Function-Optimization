import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn as nn            # containing various building blocks for your neural networks
import torch.optim as optim      # implementing various optimization algorithms
import torch.nn.functional as F  # a lower level (compared to torch.nn) interface
import math

# Local imports
import CodingFunctions
import Utils
import UtilsPlot
import Decoding


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

class Pixelwise(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Pixelwise, self).__init__()

        #################### Coding Function and Scene Parameters
        sourceExponent = 9
        ambientExponent = 6

        #### Parametrized Coding (Fourier Series)
        N = 10000
        self.N = N
        K = 3
        self.K = K
        order = 10 # The number of sinusoids to sum per function
        self.order = order
        temp = 1*torch.rand(K, order, device=device, dtype=dtype)
        self.alpha_mod = temp.clone().detach().requires_grad_(True)
        temp = 2*math.pi*torch.rand(K, order, device=device, dtype=dtype)
        self.phi_mod = temp.clone().detach().requires_grad_(True)
        temp = 1*torch.rand(K, order, device=device, dtype=dtype)
        self.alpha_demod = temp.clone().detach().requires_grad_(True)
        temp = 2*math.pi*torch.rand(K, order, device=device, dtype=dtype)
        self.phi_demod = temp.clone().detach().requires_grad_(True)
        # self.DemodFs = torch.zeros(N, K, device=device, dtype=dtype, requires_grad=True)
        # Will implement if needed, constrains to K identical, shifted demodulation functions
        # self.psi = torch.randn(K, device=device, dtype=dtype, requires_grad=True)


        #### Global parameters
        speedOfLight = 299792458. * 1000. # mm / sec 
        #### Sensor parameters
        self.T = 0.1 # Integration time. Exposure time in seconds
        self.readNoise = 20 # Standard deviation in photo-electrons
        #### Coding function parameters
        dMax = 10000 # maximum depth
        fMax = speedOfLight/(2*float(dMax)) # Maximum unambiguous repetition frequency (in Hz)
        self.tauMin = 1./fMax
        fSampling = float(dMax)*fMax # Sampling frequency of mod and demod functuion
        self.dt = self.tauMin/float(N)
        self.pAveSourcePerPixel = np.power(10, sourceExponent) # Source power. Avg number of photons emitted by the light source per second. 
        # self.pAveSourcePerPixel = pAveSource/nPixels # Avg number of photons arriving to each pixel per second. If all light is reflected back.
        freq = fMax # Fundamental frequency of modulation and demodulation functions
        self.tau = 1/freq
        #### Scene parameters
        self.pAveAmbientPerPixel = np.power(10, ambientExponent) # Ambient light power. Avg number of photons per second due to ambient light sources
        # self.pAveAmbientPerPixel = pAveAmbient/nPixels # Avg # of photons per second arriving to each pixel
        self.meanBeta = 1e-4 # Avg fraction of photons reflected from a scene points back to the detector
        #### Camera gain parameter
        ## The following bound is found by assuming the max brightness value is obtained when demod is 1. 
        self.gamma = 1./(self.meanBeta*self.T*(self.pAveAmbientPerPixel+self.pAveSourcePerPixel)) # Camera gain. Ensures all values are between 0-1.

    def forward(self, gt_depths):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        #### Calculate current coding functions based on learned parameters
        ModFs = torch.zeros(self.N, self.K, device=device, dtype=dtype, requires_grad=False)
        DemodFs = torch.zeros(self.N, self.K, device=device, dtype=dtype, requires_grad=False)
        p = torch.linspace(0, 2*math.pi, self.N)
        for k in range(0, self.K):
            for order in range(0, self.order):
                ModFs[:, k] += self.alpha_mod[k, order] * torch.sin(p*(order+1) + self.phi_mod[k, order])
                DemodFs[:, k] += self.alpha_demod[k, order] * torch.sin(p*(order+1) + self.phi_demod[k, order])
        ModFs += 0.5
        DemodFs += 0.5

        #################### Simulation
        ## Set area under the curve of outgoing ModF to the totalEnergy
        ModFs_scaled = Utils.ScaleMod(ModFs, tau=self.tauMin, pAveSource=self.pAveSourcePerPixel)
        # Calculate correlation functions (NxK matrix) and normalize it (zero mean, unit variance)
        CorrFs = Utils.GetCorrelationFunctions(ModFs_scaled,DemodFs,dt=self.dt)
        NormCorrFs = (CorrFs.t() - torch.mean(CorrFs,1)) / torch.std(CorrFs,1)
        NormCorrFs = NormCorrFs.t()
        # Compute brightness values
        BVals = Utils.ComputeBrightnessVals(ModFs=ModFs_scaled, DemodFs=DemodFs, CorrFs=CorrFs, depths=gt_depths, \
                pAmbient=self.pAveAmbientPerPixel, beta=self.meanBeta, T=self.T, tau=self.tau, dt=self.dt, gamma=self.gamma)
        
        #### Add noise
        # Calculate variance
        #noiseVar = BVals*self.gamma + math.pow(self.readNoise*self.gamma, 2) 
        # Add noise to all brightness values
        #for i in range(gt_depths.detach().numpy().size):
        #    BVals[i,:] = Utils.GetClippedBSamples(nSamples=1,BMean=BVals[i,:],BVar=noiseVar[i,:])

        decodedDepths = Decoding.DecodeXCorr(BVals,NormCorrFs)
        #print("Decoded depths: {},".format(decodedDepths))
        return decodedDepths

class ParamClipper(object):
    def __init__(self):
        None

    def __call__(self,module):
        None
        #### Clamp learned values
        #module.alpha_mod = torch.clamp(module.alpha_mod,min=0.0)
        #module.phi_mod = torch.clamp(module.phi_mod,min=0.0,max=2*math.pi)
        module.alpha_demod = torch.clamp(module.alpha_demod,min=-1.0,max=1.0)
        #module.phi_demod = torch.clamp(module.phi_demod,min=0.0,max=2*math.pi)


# Construct our model by instantiating the class defined above
model = Pixelwise()

# Instantiate Clipper
clipper = ParamClipper()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = optim.Adam([model.alpha_mod, model.alpha_demod, model.phi_mod, model.phi_demod], lr = 5e-1)


with torch.autograd.detect_anomaly():
    for t in range(100):
        # Create random Tensors to hold inputs and outputs (sample fresh each iteration (generalization))
        N = 1
        H = 10
        W = 10
        gt_depths = 1000+8000*torch.rand(N, H, W, device=device, dtype=dtype, requires_grad=True)
        #print(gt_depths)

        # Forward pass: Compute predicted y by passing x to the model
        depths_pred = model(gt_depths)

        # Compute and print loss
        loss = criterion(depths_pred, gt_depths)
        if (t%10 == 0):
            print("Iteration %d, Loss value: %f" %(t, loss.item()))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        model.apply(clipper)

print(depths_pred)

#### Calculate final coding functions based on learned parameters
ModFs = torch.zeros(model.N, model.K, device=device, dtype=dtype, requires_grad=False)
DemodFs = torch.zeros(model.N, model.K, device=device, dtype=dtype, requires_grad=False)
p = torch.linspace(0, 2*math.pi, model.N)
for k in range(0, model.K):
    for order in range(0, model.order):
        ModFs[:, k] += model.alpha_mod[k, order] * torch.sin(p*(order+1) + model.phi_mod[k, order])
        DemodFs[:, k] += model.alpha_demod[k, order] * torch.sin(p*(order+1) + model.phi_demod[k, order])

UtilsPlot.PlotCodingScheme(ModFs,DemodFs)

ModFs_np = ModFs.detach().numpy()
DemodFs_np = DemodFs.detach().numpy()
CorrFs = Utils.GetCorrelationFunctions(ModFs,DemodFs)
CorrFs_np = CorrFs.detach().numpy()
np.savez('coding_functions.npz', ModFs=ModFs_np, DemodFs=DemodFs_np, CorrFs=CorrFs_np)



