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
        super(Pixelwise, self).__init__()

        #################### Set Function Parameters
        N = 10000
        self.N = N
        K = 3
        self.K = K
        #### SUPERPOSITION OF SINUSOIDS:
        order = 4 # The number of sinusoids to sum per function
        self.order = order
        # This involves a set modulation funtion
        self.ModFs = torch.cat((2*torch.ones((int(N/2), 3), device=device, dtype=dtype), torch.zeros((int(N/2),3), device=device, dtype=dtype)),0)
        self.alpha = torch.randn(K, order, device=device, dtype=dtype, requires_grad=True)
        self.omega = torch.randn(K, order, device=device, dtype=dtype, requires_grad=True)
        self.phi = torch.randn(K, order, device=device, dtype=dtype, requires_grad=True)
        # self.DemodFs = torch.zeros(N, K, device=device, dtype=dtype, requires_grad=True)
        # Will implement if needed, constrains to K identical, shifted demodulation functions
        # self.psi = torch.randn(K, device=device, dtype=dtype, requires_grad=True)

        #### FOR RANDOM INITIALIZATION:
        # self.ModFs = torch.randn(N, K, device=device, dtype=dtype, requires_grad=True)
        # self.DemodFs = torch.randn(N, K, device=device, dtype=dtype, requires_grad=True)
        #### FOR HAMILTONIAN INITIALIZATION:
        #(ModFs_np,DemodFs_np) = CodingFunctions.GetHamK3(N = N)
        #self.ModFs = torch.tensor(ModFs_np, device=device, dtype=dtype, requires_grad=True)
        #self.DemodFs = torch.tensor(DemodFs_np, device=device, dtype=dtype, requires_grad=True)
        #### FOR COSINE INITIALIZATION:
        # (ModFs_np,DemodFs_np) = CodingFunctions.GetCosCos(N = N, K=K)
        # self.ModFs = torch.tensor(ModFs_np, device=device, dtype=dtype)
        # self.DemodFs_phase = torch.rand(1,K, dtype=dtype, device=device, requires_grad=True)
        
        #################### Coding Function and Scene Parameters
        sourceExponent = 9
        ambientExponent = 6
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
        # The following bound is found by assuming the max brightness value is obtained when demod is 1. 
        self.gamma = 1./(self.meanBeta*self.T*(self.pAveAmbientPerPixel+self.pAveSourcePerPixel)) # Camera gain. Ensures all values are between 0-1.


    def forward(self, gt_depths):
        _, H, W = gt_depths.shape
        #### Resize gt_depths to 1D
        #gt_depths = torch.reshape(gt_depths, (N, -1))
        DemodFs = torch.zeros(self.N, self.K, device=device, dtype=dtype, requires_grad=True)
        p = torch.linspace(0, self.N-1, self.N)
        for k in range(1, self.K):
            for ord in range(1, self.order):
                DemodFs[:, k] += self.alpha[k, ord] * torch.sin(self.omega[k, ord] * p + self.phi[k, ord])

        #################### Simulation
        ## Set area under the curve of outgoing ModF to the totalEnergy
        ModFs_scaled = Utils.ScaleMod(self.ModFs, tau=self.tauMin, pAveSource=self.pAveSourcePerPixel)
        # Calculate correlation functions (NxK matrix) and normalize it (zero mean, unit variance)
        # CorrFs = Utils.GetCorrelationFunctions(ModFs_scaled,self.DemodFs,dt=self.dt)
        CorrFs = Utils.GetCorrelationFunctions(ModFs_scaled,DemodFs,dt=self.dt)
        NormCorrFs = (CorrFs - torch.mean(CorrFs,0)) / torch.std(CorrFs,0)
        # BVals = Utils.ComputeBrightnessVals(ModFs=ModFs_scaled, DemodFs=self.DemodFs, depths=gt_depths, \
        #         pAmbient=self.pAveAmbientPerPixel, beta=self.meanBeta, T=self.T, tau=self.tau, dt=self.dt, gamma=self.gamma)
        BVals = Utils.ComputeBrightnessVals(ModFs=ModFs_scaled, DemodFs=DemodFs, depths=gt_depths, \
                pAmbient=self.pAveAmbientPerPixel, beta=self.meanBeta, T=self.T, tau=self.tau, dt=self.dt, gamma=self.gamma)
        # print("BVals shape:", BVals.shape)
        #### Add noise
        # Calculate variance
        #noiseVar = BVals*self.gamma + math.pow(self.readNoise*self.gamma, 2) 
        # Add noise to all brightness values
        #for i in range(gt_depths.detach().numpy().size):
        #    BVals[i,:] = Utils.GetClippedBSamples(nSamples=1,BMean=BVals[i,:],BVar=noiseVar[i,:])

        decodedDepths = Decoding.DecodeXCorr(BVals,NormCorrFs)
        #print("Decoded depths: {},".format(decodedDepths))
        return decodedDepths

# Create random Tensors to hold inputs and outputs
N = 1
H = 10
W = 10
#gt_depths = 10*torch.ones(N, H, W, device=device, dtype=dtype, requires_grad=True)
gt_depths = 9000*torch.rand(N, H, W, device=device, dtype=dtype, requires_grad=True)
print("Ground Truth Depths:", gt_depths)
print("Ground Truth Depths Shape:", gt_depths.shape)

gt_depths_init = gt_depths.clone()
# y = torch.randn(1, 1, device=device, dtype=dtype, requires_grad=True)

# Construct our model by instantiating the class defined above
model = Pixelwise()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
# criterion = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = optim.Adam([model.ModFs, model.DemodFs], lr = 1e-6)
optimizer = optim.Adam([model.alpha, model.omega, model.phi], lr = 1e-6)

with torch.autograd.detect_anomaly():
    for t in range(1, 100 + 1):
        # Forward pass: Compute predicted y by passing x to the model
        depths_pred = model(gt_depths)

        # Compute and print loss
        #loss = criterion(depths_pred, 1000*torch.ones([1,2,2,3], dtype=torch.float, device=device, requires_grad=True))
        # loss = criterion(depths_pred, goal)
        loss = criterion(depths_pred, gt_depths)

        if (t == 1 or t % 10 == 0):
            print("Iteration %d, Loss: %f" %(t, loss.item()))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    print("Predicted Depths:", depths_pred)
    ModFs_scaled = Utils.ScaleMod(model.ModFs, tau=model.tauMin, pAveSource=model.pAveSourcePerPixel)
    print("Scaled Modulation Functions:", ModFs_scaled)

    # UtilsPlot.PlotCodingScheme(model.ModFs,model.DemodFs)
