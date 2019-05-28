import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn as nn            # containing various building blocks for your neural networks
import torch.optim as optim      # implementing various optimization algorithms
import torch.nn.functional as F  # a lower level (compared to torch.nn) interface
import math
from scipy.io import loadmat
import random

# Local imports
import CodingFunctions
import Utils
import UtilsPlot
import Decoding


dtype = torch.float
device = torch.device("cpu")
use_gpu = True
if use_gpu:
    device = torch.device("cuda:0") # Uncomment this to run on GPU

class CNN(torch.nn.Module):
    def __init__(self, architecture):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(CNN, self).__init__()

        #################### Coding Function and Scene Parameters
        sourceExponent = 9
        ambientExponent = 6

        #### Coding (Fix ModFs to square, initialize DemodFs at random)
        #N = 10000
        #K = 3
        #self.ModFs = torch.cat((2*torch.ones((int(N/2), 3), device=device, dtype=dtype), torch.zeros((int(N/2),3), device=device, dtype=dtype)),0)
        #temp = 1/np.power(10, 0) * torch.rand(N, K, device=device, dtype=dtype, requires_grad=True) # scaled random initialization
        #self.DemodFs = temp.clone().detach().requires_grad_(True)

        #### Coding (Initialize ModFs and DemodFs at random)
        #N = 10000
        #K = 3
        #temp = 1/np.power(10, 5) * torch.rand(N, K, device=device, dtype=dtype, requires_grad=True) # scaled random initialization
        #self.ModFs = temp.clone().detach().requires_grad_(True)
        #temp = 1/np.power(10, 5) * torch.rand(N, K, device=device, dtype=dtype, requires_grad=True) # scaled random initialization
        #self.DemodFs = temp.clone().detach().requires_grad_(True)

        #### Coding (Initialize at Hamiltonian)
        N = 10000
        K = 3
        self.K = K
        (ModFs_np,DemodFs_np) = CodingFunctions.GetHamK3(N = N)
        temp = torch.tensor(ModFs_np, device=device, dtype=dtype)
        self.ModFs = temp.clone().detach().requires_grad_(True).to(device=device)
        temp = torch.tensor(DemodFs_np, device=device, dtype=dtype)
        self.DemodFs = temp.clone().detach().requires_grad_(True).to(device=device)

        self.architecture = architecture
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

        #### CNN Initialization
        CNN.network(self, architecture, None, True)
        
        
    def forward(self, gt_depths):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        #################### Simulation
        ## Set area under the curve of outgoing ModF to the totalEnergy
        ModFs_scaled = Utils.ScaleMod(self.ModFs, device=device, tau=self.tauMin, pAveSource=self.pAveSourcePerPixel)
        # Clip the demodulation functions to [0,1]
        #DemodFs_clipped = torch.clamp(self.DemodFs, 0.0, 1.0)
        # Calculate correlation functions (NxK matrix) and normalize it (zero mean, unit variance)
        CorrFs = Utils.GetCorrelationFunctions(ModFs_scaled,self.DemodFs,device=device,dt=self.dt)
        NormCorrFs = (CorrFs.t() - torch.mean(CorrFs,1)) / torch.std(CorrFs,1)
        NormCorrFs = NormCorrFs.t()
        # Compute brightness values
        BVals = Utils.ComputeBrightnessVals(ModFs=ModFs_scaled, DemodFs=self.DemodFs, CorrFs=CorrFs, depths=gt_depths, \
                pAmbient=self.pAveAmbientPerPixel, beta=self.meanBeta, T=self.T, tau=self.tau, dt=self.dt, gamma=self.gamma)
        BVals = BVals.permute(0,3,1,2) # Put channel dimension at right position

        #### Add noise
        # Calculate variance
        #noiseVar = BVals*self.gamma + math.pow(self.readNoise*self.gamma, 2) 
        # Add noise to all brightness values
        #for i in range(gt_depths.detach().numpy().size):
        #    BVals[i,:] = Utils.GetClippedBSamples(nSamples=1,BMean=BVals[i,:],BVar=noiseVar[i,:])


        # Normalize BVals
        BVals_mean = torch.mean(BVals)
        BVals_std = torch.std(BVals)
        BVals = (BVals - BVals_mean)/BVals_std

        #### CNN Network
        out = CNN.network(self, self.architecture, BVals)

        decodedDepths = torch.squeeze(out, 1) # Remove channel dimension
        return decodedDepths

    def network(self, architecture, BVals=None, init=False):
        if architecture == 'sequential':
            if init == True:
                self.layer_down1 = nn.Sequential(
                    nn.Conv2d(self.K, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU())
                    #nn.MaxPool2d(kernel_size=2, stride=2))
                self.layer_down2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU())
                    #nn.MaxPool2d(kernel_size=2, stride=2))
                self.layer_down3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU())
                    #nn.MaxPool2d(kernel_size=2, stride=2))

                self.layer_same1 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU())

                self.layer_up1 = nn.Sequential(
                    #nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.BatchNorm2d(128),
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                    nn.ReLU())
                self.layer_up2 = nn.Sequential(
                    #nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.BatchNorm2d(64),
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU())
                self.layer_up3 = nn.Sequential(
                    #nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.BatchNorm2d(32),
                    nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1))
                    #nn.Tanh())
            else:
                # Down Convolution
                x = self.layer_down1(BVals)
                x = self.layer_down2(x)
                x = self.layer_down3(x)
                # Same size Convolution
                x = self.layer_same1(x)
                # Up Convolution
                x = self.layer_up1(x)
                x = self.layer_up2(x)
                x = self.layer_up3(x)
                return x

        if architecture == 'skip_connection':
            if init == True:
                self.layer_down1 = nn.Sequential(
                    nn.Conv2d(self.K, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU())
                    #nn.MaxPool2d(kernel_size=2, stride=2))
                self.layer_down2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU())
                    #nn.MaxPool2d(kernel_size=2, stride=2))
                self.layer_down3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU())
                    #nn.MaxPool2d(kernel_size=2, stride=2))

                self.layer_same1 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU())

                self.layer_up1a = nn.Sequential(
                    #nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.BatchNorm2d(128))
                self.layer_up1b = nn.Sequential(
                    nn.ConvTranspose2d(2*128, 64, kernel_size=4, stride=2, padding=1),
                    nn.ReLU())
                self.layer_up2a = nn.Sequential(
                    #nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.BatchNorm2d(64))
                self.layer_up2b = nn.Sequential(
                    nn.ConvTranspose2d(2*64, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU())
                self.layer_up3a = nn.Sequential(
                    #nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.BatchNorm2d(32))
                self.layer_up3b = nn.Sequential(
                    nn.ConvTranspose2d(2*32, 1, kernel_size=4, stride=2, padding=1))
                    #nn.Tanh())
            else:
                # Down Convolution
                x1 = self.layer_down1(BVals)
                x2 = self.layer_down2(x1)
                x3 = self.layer_down3(x2)
                # Same size Convolution
                x = self.layer_same1(x3)
                # Up Convolution
                x = self.layer_up1a(x)
                x = self.layer_up1b(torch.cat([x, x3], 1))
                x = self.layer_up2a(x)
                x = self.layer_up2b(torch.cat([x, x2], 1))
                x = self.layer_up3a(x)
                x = self.layer_up3b(torch.cat([x, x1], 1))
                return x

        if architecture == 'deep_skip_connection':
            if init == True:
                self.layer_down1 = nn.Sequential(
                    nn.Conv2d(self.K, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU())
                    #nn.MaxPool2d(kernel_size=2, stride=2))
                self.layer_down2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU())
                    #nn.MaxPool2d(kernel_size=2, stride=2))
                self.layer_down3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU())
                    #nn.MaxPool2d(kernel_size=2, stride=2))
                self.layer_down4 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU())
                    #nn.MaxPool2d(kernel_size=2, stride=2))
                self.layer_down5 = nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU())
                    #nn.MaxPool2d(kernel_size=2, stride=2))

                self.layer_same1 = nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU())
                self.layer_same2 = nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU())

                self.layer_up1a = nn.Sequential(
                    #nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.BatchNorm2d(512))
                self.layer_up1b = nn.Sequential(
                    nn.ConvTranspose2d(2*512, 256, kernel_size=4, stride=2, padding=1),
                    nn.ReLU())
                self.layer_up2a = nn.Sequential(
                    #nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.BatchNorm2d(256))
                self.layer_up2b = nn.Sequential(
                    nn.ConvTranspose2d(2*256, 128, kernel_size=4, stride=2, padding=1),
                    nn.ReLU())
                self.layer_up3a = nn.Sequential(
                    #nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.BatchNorm2d(128))
                self.layer_up3b = nn.Sequential(
                    nn.ConvTranspose2d(2*128, 64, kernel_size=4, stride=2, padding=1),
                    nn.ReLU())
                self.layer_up4a = nn.Sequential(
                    #nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.BatchNorm2d(64))
                self.layer_up4b = nn.Sequential(
                    nn.ConvTranspose2d(2*64, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU())
                self.layer_up5a = nn.Sequential(
                    #nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.BatchNorm2d(32))
                self.layer_up5b = nn.Sequential(
                    nn.ConvTranspose2d(2*32, 1, kernel_size=4, stride=2, padding=1))
                    #nn.Tanh())
            else:
                # Down Convolution
                x1 = self.layer_down1(BVals)
                x2 = self.layer_down2(x1)
                x3 = self.layer_down3(x2)
                x4 = self.layer_down4(x3)
                x5 = self.layer_down5(x4)
                # Same size Convolution
                x = self.layer_same1(x5)
                x = self.layer_same2(x)
                # Up Convolution
                x = self.layer_up1a(x)
                x = self.layer_up1b(torch.cat([x, x5], 1))
                x = self.layer_up2a(x)
                x = self.layer_up2b(torch.cat([x, x4], 1))
                x = self.layer_up3a(x)
                x = self.layer_up3b(torch.cat([x, x3], 1))
                x = self.layer_up4a(x)
                x = self.layer_up4b(torch.cat([x, x2], 1))
                x = self.layer_up5a(x)
                x = self.layer_up5b(torch.cat([x, x1], 1))
                return x


# Construct our model by instantiating the class defined above
# Choose from: 'sequential', 'skip_connection'
model = CNN('deep_skip_connection')
if use_gpu:
    model.cuda()
print("MODEL MADE")
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr = 1e-4)

# Load data
data = loadmat('patches_64.mat')
train = data['patches_train']
val = data['patches_val']
test = data['patches_test']

train = torch.from_numpy(train[:50000,:,:])
val = torch.from_numpy(val)
test = torch.from_numpy(test)

train_gt_depths = train.float().to(device).requires_grad_(True)
val_gt_depths = val.float().to(device).requires_grad_(True)
test_gt_depths = test.float().to(device).requires_grad_(True)

train_gt_depths_mean = torch.mean(train_gt_depths)
val_gt_depths_mean = torch.mean(val_gt_depths)
test_gt_depths_mean = torch.mean(test_gt_depths)
print("Train mean:", train_gt_depths_mean.item())
print("Val mean:", val_gt_depths_mean.item())

train_gt_depths_std = torch.std(train_gt_depths)
val_gt_depths_std = torch.std(val_gt_depths)
test_gt_depths_std = torch.std(test_gt_depths)
print("Train std:", train_gt_depths_std.item())
print("Val std:", val_gt_depths_std.item())

train_normalized_gt_depths = (train_gt_depths-train_gt_depths_mean)/train_gt_depths_std
val_normalized_gt_depths = (val_gt_depths-val_gt_depths_mean)/val_gt_depths_std
test_normalized_gt_depths = (test_gt_depths-test_gt_depths_mean)/test_gt_depths_std

print("DATA IMPORTED")

with torch.autograd.detect_anomaly():
    iteration = 0
    increased = 0
    patience = 1000
    train_batch_size = 8
    val_batch_size = 1
    val_every = 100
    val_number = val_gt_depths.shape[0]
    train_enumeration = torch.arange(train_gt_depths.shape[0])
    train_enumeration = train_enumeration.tolist()

    while increased <= patience:
        train_ind = random.sample(train_enumeration, train_batch_size)
        # Forward pass: Compute predicted y by passing x to the model
        train_depths_pred = model(train_gt_depths[train_ind])
        # Compute and print loss
        train_loss = criterion(train_depths_pred, train_normalized_gt_depths[train_ind])
        train_depths_pred_unnorm = train_depths_pred*train_gt_depths_std+train_gt_depths_mean
        train_MSE = criterion(train_depths_pred_unnorm, train_gt_depths[train_ind])        
        iteration = iteration + 1

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        train_loss.backward(retain_graph=True)
        optimizer.step()
        if use_gpu:
            #print("Memory before freeing cache:", torch.cuda.memory_allocated(device))
            torch.cuda.empty_cache()
            #print("Memory after freeing cache: ", torch.cuda.memory_allocated(device))

        if iteration == 1 or iteration%val_every == 0:
            with torch.no_grad():
                val_loss = 0
                val_MSE = 0
                for b in range(val_number):
                    if b + val_batch_size < val_number:
                        val_depths_pred = model(val_gt_depths[b:b+val_batch_size])
                        val_loss = val_loss + criterion(val_depths_pred, val_normalized_gt_depths[b:b+val_batch_size])
                        val_depths_pred_unnorm = val_depths_pred*val_gt_depths_std+val_gt_depths_mean
                        val_MSE = val_MSE + criterion(val_depths_pred_unnorm, val_gt_depths[b:b+val_batch_size])
                    else:
                        val_depths_pred = model(val_gt_depths[b:])
                        val_loss = val_loss + criterion(val_depths_pred, val_normalized_gt_depths[b:])
                        val_depths_pred_unnorm = val_depths_pred*val_gt_depths_std+val_gt_depths_mean
                        val_MSE = val_MSE + criterion(val_depths_pred_unnorm, val_gt_depths[b:])
                    b = b + val_batch_size
                    if use_gpu:
                        torch.cuda.empty_cache()
                val_loss /= val_number # mean loss
                val_MSE /= val_number # mean MSE 

                print("Iteration: %d, Train Loss: %f, Val Loss: %f" %(iteration, train_loss.item(), val_loss.item()))
                # Unnormalize and output MSE loss (for interpretability)
                print("Train MSE: %f, Val MSE: %f" %(train_MSE,val_MSE))
                if iteration == 1 or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_iteration = iteration
                    best_model = model
                    increased = 0
                else:
                    increased = increased + 1

print("DONE TRAINING")
print("Best Validation Loss:", best_val_loss.item())
print("Best Iteration:", best_iteration)

with torch.no_grad():
    test_loss = 0
    test_MSE = 0
    test_batch_size = 1
    test_number = test_gt_depths.shape[0]
    for b in range(test_number):
        if b + test_batch_size < test_number:
            test_depths_pred = best_model(test_gt_depths[b:b+test_batch_size])
            test_loss = test_loss + criterion(test_depths_pred, test_normalized_gt_depths[b:b+test_batch_size])
            test_depths_pred_unnorm = test_depths_pred*test_gt_depths_std+test_gt_depths_mean
            test_MSE = test_MSE + criterion(test_depths_pred_unnorm, test_gt_depths[b:b+test_batch_size])
        else:
            test_depths_pred = best_model(test_gt_depths[b:])
            test_loss = test_loss + criterion(test_depths_pred, test_normalized_gt_depths[b:])
            test_depths_pred_unnorm = test_depths_pred*test_gt_depths_std+test_gt_depths_mean
            test_MSE = test_MSE + criterion(test_depths_pred_unnorm, test_gt_depths[b:])
        b = b + test_batch_size
        if use_gpu:
            torch.cuda.empty_cache()
    test_loss /= test_number # mean loss
    test_MSE /= test_number # mean MSE

print("Test Loss: %f, Test MSE: %f" %(test_loss.item(), test_MSE))

#ModFs_np = best_model.ModFs.cpu().detach().numpy()
#DemodFs_np = best_model.DemodFs.cpu().detach().numpy()
#CorrFs = Utils.GetCorrelationFunctions(best_model.ModFs,best_model.DemodFs, device=device)
#CorrFs_np = CorrFs.cpu().detach().numpy()
#np.savez('coding_functions.npz', ModFs=ModFs_np, DemodFs=DemodFs_np, CorrFs=CorrFs_np)
