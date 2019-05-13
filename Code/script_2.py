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

# Using this tutorial:
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# print("torch version:", torch.__version__)


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

        #################### Set Function Parameters
        #N = 10000
        #K = 3
        #(ModFs_np,DemodFs_np) = CodingFunctions.GetHamK3(N = N)
        #self.ModFs = torch.tensor(ModFs_np, device=device, dtype=dtype, requires_grad=True)
        #self.DemodFs = torch.tensor(DemodFs_np, device=device, dtype=dtype, requires_grad=True)

        N = 10000
        K = 3
        (ModFs_np,DemodFs_np) = CodingFunctions.GetCosCos(N = N, K=K)
        self.ModFs = torch.tensor(ModFs_np, device=device, dtype=dtype)
        self.DemodFs_phase = torch.rand(1,K, dtype=dtype, device=device, requires_grad=True)
        n_lin = torch.linspace(0, 1, steps=10000)
        t1 = 0.5+0.5*torch.sin(2*math.pi*(n_lin - self.DemodFs_phase[0,0]))
        t2 = 0.5+0.5*torch.sin(2*math.pi*(n_lin - self.DemodFs_phase[0,1]))
        t3 = 0.5+0.5*torch.sin(2*math.pi*(n_lin - self.DemodFs_phase[0,2]))
        print(t1.shape)
        self.DemodFs = torch.cat((torch.reshape(t1,[-1,1]),torch.reshape(t2,[-1,1]),torch.reshape(t3,[-1,1])),dim=1)
        print(self.DemodFs.shape)
        #e1_K1 = self.DemodFs_edges[0,0].long() if self.DemodFs_edges[0,0] < self.DemodFs_edges[1,0] else self.DemodFs_edges[1,0].long()
        #e2_K1 = self.DemodFs_edges[1,0].long() if self.DemodFs_edges[0,0] < self.DemodFs_edges[1,0] else self.DemodFs_edges[0,0].long()
        #e1_K2 = self.DemodFs_edges[0,1].long() if self.DemodFs_edges[0,1] < self.DemodFs_edges[1,1] else self.DemodFs_edges[1,1].long()
        #e2_K2 = self.DemodFs_edges[1,1].long() if self.DemodFs_edges[0,1] < self.DemodFs_edges[1,1] else self.DemodFs_edges[0,1].long()
        #e1_K3 = self.DemodFs_edges[0,2].long() if self.DemodFs_edges[0,2] < self.DemodFs_edges[1,2] else self.DemodFs_edges[1,2].long()
        #e2_K3 = self.DemodFs_edges[1,2].long() if self.DemodFs_edges[0,2] < self.DemodFs_edges[1,2] else self.DemodFs_edges[0,2].long()
        #t1 = torch.cat((torch.zeros((e1_K1,1)),torch.ones((e2_K1-e1_K1,1)),torch.zeros((10000-e2_K1,1))),dim=0)
        #t2 = torch.cat((torch.zeros((e1_K2,1)),torch.ones((e2_K2-e1_K2,1)),torch.zeros((10000-e2_K2,1))),dim=0)
        #t3 = torch.cat((torch.zeros((e1_K3,1)),torch.ones((e2_K3-e1_K3,1)),torch.zeros((10000-e2_K3,1))),dim=0)
        #self.DemodFs = torch.cat((t1,t2,t3),dim=1)

        #N = 10000
        #K = 3
        #(ModFs_np,DemodFs_np) = CodingFunctions.GetHamK3(N = N)
        #self.ModFs = torch.tensor(ModFs_np, device=device, dtype=dtype)
        #self.DemodFs_edges = torch.randint(0,10000,(2,K), dtype=dtype, device=device, requires_grad=True)
        #e1_K1 = self.DemodFs_edges[0,0].long() if self.DemodFs_edges[0,0] < self.DemodFs_edges[1,0] else self.DemodFs_edges[1,0].long()
        #e2_K1 = self.DemodFs_edges[1,0].long() if self.DemodFs_edges[0,0] < self.DemodFs_edges[1,0] else self.DemodFs_edges[0,0].long()
        #e1_K2 = self.DemodFs_edges[0,1].long() if self.DemodFs_edges[0,1] < self.DemodFs_edges[1,1] else self.DemodFs_edges[1,1].long()
        #e2_K2 = self.DemodFs_edges[1,1].long() if self.DemodFs_edges[0,1] < self.DemodFs_edges[1,1] else self.DemodFs_edges[0,1].long()
        #e1_K3 = self.DemodFs_edges[0,2].long() if self.DemodFs_edges[0,2] < self.DemodFs_edges[1,2] else self.DemodFs_edges[1,2].long()
        #e2_K3 = self.DemodFs_edges[1,2].long() if self.DemodFs_edges[0,2] < self.DemodFs_edges[1,2] else self.DemodFs_edges[0,2].long()
        #t1 = torch.cat((torch.zeros((e1_K1,1)),torch.ones((e2_K1-e1_K1,1)),torch.zeros((10000-e2_K1,1))),dim=0)
        #t2 = torch.cat((torch.zeros((e1_K2,1)),torch.ones((e2_K2-e1_K2,1)),torch.zeros((10000-e2_K2,1))),dim=0)
        #t3 = torch.cat((torch.zeros((e1_K3,1)),torch.ones((e2_K3-e1_K3,1)),torch.zeros((10000-e2_K3,1))),dim=0)
        #self.DemodFs = torch.cat((t1,t2,t3),dim=1)

        #self.ModFs = torch.randn(N, K, device=device, dtype=dtype, requires_grad=True)
        #self.DemodFs = torch.randn(N, K, device=device, dtype=dtype, requires_grad=True)

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
        ## The following bound is found by assuming the max brightness value is obtained when demod is 1. 
        self.gamma = 1./(self.meanBeta*self.T*(self.pAveAmbientPerPixel+self.pAveSourcePerPixel)) # Camera gain. Ensures all values are between 0-1.

    def forward(self, gt_depths):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ### Resize gt_depths to 1D
        N, H, W = gt_depths.shape
        #gt_depths = torch.reshape(gt_depths, (N, -1))

        #################### Simulation
        ## Set area under the curve of outgoing ModF to the totalEnergy
        ModFs_scaled = Utils.ScaleMod(self.ModFs, tau=self.tauMin, pAveSource=self.pAveSourcePerPixel)

        # Calculate correlation functions (NxK matrix) and normalize it (zero mean, unit variance)
        CorrFs = Utils.GetCorrelationFunctions(ModFs_scaled,self.DemodFs,dt=self.dt)
        NormCorrFs = (CorrFs - torch.mean(CorrFs,0)) / torch.std(CorrFs,0)

        BVals = Utils.ComputeBrightnessVals(ModFs=self.ModFs, DemodFs=self.DemodFs, depths=gt_depths, \
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

        # decodedDepths = ... reshape
        #return ModFs_scaled
        # return CorrFs
        #return NormCorrFs
        #return BVals

# Create random Tensors to hold inputs and outputs
N = 1
H = 1
W = 1
#gt_depths = 10*torch.ones(N, H, W, device=device, dtype=dtype, requires_grad=True)
gt_depths = 9000*torch.rand(N, H, W, device=device, dtype=dtype, requires_grad=True)
print('gt_depths:',gt_depths)

gt_depths_init = gt_depths.clone()
# y = torch.randn(1, 1, device=device, dtype=dtype, requires_grad=True)

# Construct our model by instantiating the class defined above
model = Pixelwise()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam([model.DemodFs_phase], lr = 1e-3)
#optimizer = optim.Adam([model.ModFs, model.DemodFs], lr = 1e-6)


# Goal correlation function (build a triangular function)
# tr = torch.linspace(0,1,steps=25,dtype=torch.float)
# tr = tr.reshape(-1,1)
# tr = torch.cat((tr,tr,tr),1)
# tf = torch.linspace(1,0,steps=25,dtype=torch.float)
# tf = tf.reshape(-1,1)
# tf = torch.cat((tf,tf,tf),1)
# goal = torch.tensor(torch.cat((tr,tf,tf-1,tr-1),0), device=device, requires_grad=True)
# print(torch.mean(goal,0))
# print(torch.std(goal,0))
# print(goal)



with torch.autograd.detect_anomaly():
    for t in range(100):
        # Forward pass: Compute predicted y by passing x to the model
        depths_pred = model(gt_depths)

        # Compute and print loss
        #loss = criterion(depths_pred, 1000*torch.ones([1,2,2,3], dtype=torch.float, device=device, requires_grad=True))
        # loss = criterion(depths_pred, goal)
        loss = criterion(depths_pred, gt_depths)

        if (t%10 == 0):
            print("Iteration %d, Loss value: %f" %(t, loss.item()))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    print(depths_pred)
    ModFs_scaled = Utils.ScaleMod(model.ModFs, tau=model.tauMin, pAveSource=model.pAveSourcePerPixel)
    print(ModFs_scaled)

    UtilsPlot.PlotCodingScheme(model.ModFs,model.DemodFs)
    

