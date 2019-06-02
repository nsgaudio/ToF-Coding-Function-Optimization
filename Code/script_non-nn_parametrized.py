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
#device = torch.device("cuda:0") # Uncomment this to run on GPU

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
        order = 20 # The number of sinusoids to sum per function
        self.order = order

        # Initialize at Sinusoid
        #temp_alpha_mod = torch.zeros(1, order, device=device, dtype=dtype)
        #temp_alpha_demod = torch.zeros(K, order, device=device, dtype=dtype)
        #temp_phi_mod = torch.zeros(1, order, device=device, dtype=dtype)
        #temp_phi_demod = torch.zeros(K, order, device=device, dtype=dtype)
        #temp_alpha_mod[0,0] = 1
        #temp_alpha_demod[:,0] = torch.ones(K, device=device, dtype=dtype)
        #for i in range(self.K):
        #    temp_phi_demod[i,:] = -i*2/3*math.pi * torch.ones(order, device=device, dtype=dtype)
        #self.alpha_mod = temp_alpha_mod.clone().detach().requires_grad_(True)
        #self.alpha_demod = temp_alpha_demod.clone().detach().requires_grad_(True)
        #self.phi_mod = temp_phi_mod.clone().detach().requires_grad_(True)
        #self.phi_demod = temp_phi_demod.clone().detach().requires_grad_(True)

        # Initialize at Square
        temp_alpha_mod = torch.zeros(1, order, device=device, dtype=dtype)
        temp_alpha_demod = torch.zeros(K, order, device=device, dtype=dtype)
        temp_phi_mod = torch.zeros(1, order, device=device, dtype=dtype)
        temp_phi_demod = torch.zeros(K, order, device=device, dtype=dtype)
        for i in range(self.order):
            temp_alpha_mod[0,i] = 2*2/((i+1)*math.pi) * np.sin((i+1)*math.pi/2) * torch.ones(1, device=device, dtype=dtype)
            temp_alpha_demod[:,i] = 2*1/((i+1)*math.pi) * np.sin((i+1)*math.pi/2) * torch.ones(K, device=device, dtype=dtype)
        #temp_phi_mod[0,:] = -1/12*math.pi * torch.ones(order, device=device, dtype=dtype)
        for i in range(self.K):
            temp_phi_demod[i,:] = i*2/3*math.pi * torch.ones(order, device=device, dtype=dtype)
        self.alpha_mod = temp_alpha_mod.clone().detach().requires_grad_(True)
        self.alpha_demod = temp_alpha_demod.clone().detach().requires_grad_(True)
        self.phi_mod = temp_phi_mod.clone().detach().requires_grad_(True)
        self.phi_demod = temp_phi_demod.clone().detach().requires_grad_(True)      

        # Initialize at Hamiltonian
        #temp_alpha_mod = torch.zeros(1, order, device=device, dtype=dtype)
        #temp_alpha_demod = torch.zeros(K, order, device=device, dtype=dtype)
        #temp_phi_mod = torch.zeros(1, order, device=device, dtype=dtype)
        #temp_phi_demod = torch.zeros(K, order, device=device, dtype=dtype)
        #for i in range(self.order):
        #    temp_alpha_mod[0,i] = 2*6/((i+1)*math.pi) * np.sin((i+1)*math.pi/6) * torch.ones(1, device=device, dtype=dtype)
        #    temp_alpha_demod[:,i] = 2*1/((i+1)*math.pi) * np.sin((i+1)*math.pi/2) * torch.ones(K, device=device, dtype=dtype)
        #temp_phi_mod[0,:] = -1/12*math.pi * torch.ones(order, device=device, dtype=dtype)
        #for i in range(self.K):
        #    temp_phi_demod[i,:] = i*2/3*math.pi * torch.ones(order, device=device, dtype=dtype)
        #self.alpha_mod = temp_alpha_mod.clone().detach().requires_grad_(True)
        #self.alpha_demod = temp_alpha_demod.clone().detach().requires_grad_(True)
        #self.phi_mod = temp_phi_mod.clone().detach().requires_grad_(True)
        #self.phi_demod = temp_phi_demod.clone().detach().requires_grad_(True)

        # Random initialization
        #temp = 1*torch.rand(K, order, device=device, dtype=dtype)
        #self.alpha_mod = temp.clone().detach().requires_grad_(True)
        #temp = 2*math.pi*torch.rand(K, order, device=device, dtype=dtype)
        #self.phi_mod = temp.clone().detach().requires_grad_(True)
        #temp = 1*torch.rand(K, order, device=device, dtype=dtype)
        #self.alpha_demod = temp.clone().detach().requires_grad_(True)
        #temp = 2*math.pi*torch.rand(K, order, device=device, dtype=dtype)
        #self.phi_demod = temp.clone().detach().requires_grad_(True)


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
        ModFs_func = torch.zeros(self.N, 1, device=device, dtype=dtype, requires_grad=False)
        DemodFs_func = torch.zeros(self.N, self.K, device=device, dtype=dtype, requires_grad=False)
        ModFs = torch.zeros(self.N, 1, device=device, dtype=dtype, requires_grad=False)
        DemodFs = torch.zeros(self.N, self.K, device=device, dtype=dtype, requires_grad=False)
        p = torch.linspace(0, 2*math.pi, self.N, device=device)
        for order in range(0, self.order):
            ModFs_func[:, 0] += self.alpha_mod[0, order] * torch.cos((p+self.phi_mod[0, order])*(order+1))
            for k in range(0, self.K):
                DemodFs_func[:, k] += self.alpha_demod[k, order] * torch.cos((p+self.phi_demod[k, order])*(order+1))
            
        # Normalize ModFs and DemodFs
        min_ModFs_func, _ = torch.min(ModFs_func, dim=0)
        ModFs = ModFs_func - min_ModFs_func # ModFs can't be lower than zero (negative light)
        min_DemodFs_func, _ = torch.min(DemodFs_func, dim=0)
        max_DemodFs_func, _ = torch.max(DemodFs_func, dim=0)
        DemodFs = (DemodFs_func - min_DemodFs_func) / (max_DemodFs_func - min_DemodFs_func) # DemodFs can only be 0->1

        #################### Simulation
        ## Set area under the curve of outgoing ModF to the totalEnergy
        ModFs = torch.cat((ModFs,ModFs,ModFs),dim=1)
        ModFs_scaled = Utils.ScaleMod(ModFs, device, tau=self.tauMin, pAveSource=self.pAveSourcePerPixel)
        # Calculate correlation functions (NxK matrix) and normalize it (zero mean, unit variance)
        CorrFs = Utils.GetCorrelationFunctions(ModFs_scaled,DemodFs,device,dt=self.dt)
        NormCorrFs = (CorrFs.t() - torch.mean(CorrFs,1)) / torch.std(CorrFs,1)
        NormCorrFs = NormCorrFs.t()
        # Compute brightness values
        BVals = Utils.ComputeBrightnessVals(ModFs=ModFs_scaled, DemodFs=DemodFs, CorrFs=CorrFs, depths=gt_depths, \
                pAmbient=self.pAveAmbientPerPixel, beta=self.meanBeta, T=self.T, tau=self.tau, dt=self.dt, gamma=self.gamma)
        
        #### Add noise
        # Calculate variance
        noiseVar = BVals*self.gamma + math.pow(self.readNoise*self.gamma, 2) 
        # Add noise to all brightness values
        BVals = Utils.GetClippedBSamples(nSamples=1,BMean=BVals,BVar=noiseVar,device=device)

        decodedDepths = Decoding.DecodeXCorr(BVals,NormCorrFs,device)
        #print("Decoded depths: {},".format(decodedDepths))
        return decodedDepths


# Construct our model by instantiating the class defined above
model = Pixelwise()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam([model.alpha_mod, model.alpha_demod, model.phi_mod, model.phi_demod], lr = 5e-3)


with torch.autograd.detect_anomaly():
    best_loss = 1e6
    for t in range(1000):
        # Create random Tensors to hold inputs and outputs (sample fresh each iteration (generalization))
        N = 1
        H = 20
        W = 20
        gt_depths = 1000+8000*torch.rand(N, H, W, device=device, dtype=dtype, requires_grad=True)
        #print(gt_depths)

        # Forward pass: Compute predicted y by passing x to the model
        depths_pred = model(gt_depths)

        # Compute and print loss
        loss = criterion(depths_pred, gt_depths)
        if (t%10 == 0):
            print("Iteration %d, Loss value: %f" %(t, loss.item()))

        # Save best model
        if loss < best_loss:
            best_loss = loss
            best_model = model

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Clamp parameters to physically possible values
        #model.alpha_demod.data.clamp_(min=0,max=1)

model = best_model

#### Calculate current coding functions based on learned parameters
ModFs_func = torch.zeros(model.N, 1, device=device, dtype=dtype, requires_grad=False)
DemodFs_func = torch.zeros(model.N, model.K, device=device, dtype=dtype, requires_grad=False)
ModFs = torch.zeros(model.N, 1, device=device, dtype=dtype, requires_grad=False)
DemodFs = torch.zeros(model.N, model.K, device=device, dtype=dtype, requires_grad=False)
p = torch.linspace(0, 2*math.pi, model.N, device=device)
for order in range(0, model.order):
    ModFs_func[:, 0] += model.alpha_mod[0, order] * torch.cos((p+model.phi_mod[0, order])*(order+1))
    for k in range(0, model.K):
        DemodFs_func[:, k] += model.alpha_demod[k, order] * torch.cos((p+model.phi_demod[k, order])*(order+1))
    
# Normalize ModFs and DemodFs
min_ModFs_func, _ = torch.min(ModFs_func, dim=0)
ModFs = ModFs_func - min_ModFs_func # ModFs can't be lower than zero (negative light)
min_DemodFs_func, _ = torch.min(DemodFs_func, dim=0)
max_DemodFs_func, _ = torch.max(DemodFs_func, dim=0)
DemodFs = (DemodFs_func - min_DemodFs_func) / (max_DemodFs_func - min_DemodFs_func) # DemodFs can only be 0->1
ModFs = torch.cat((ModFs,ModFs,ModFs),dim=1)

UtilsPlot.PlotCodingScheme(ModFs,DemodFs,device)

ModFs_np = ModFs.cpu().detach().numpy()
DemodFs_np = DemodFs.cpu().detach().numpy()
CorrFs = Utils.GetCorrelationFunctions(ModFs,DemodFs,device)
CorrFs_np = CorrFs.cpu().detach().numpy()
np.savez('coding_functions/param_initSquare.npz', ModFs=ModFs_np, DemodFs=DemodFs_np, CorrFs=CorrFs_np)

