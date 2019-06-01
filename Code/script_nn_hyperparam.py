import numpy as np
import torch
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn as nn            # containing various building blocks for your neural networks
import torch.optim as optim      # implementing various optimization algorithms
import torch.nn.functional as F  # a lower level (compared to torch.nn) interface
import math
from scipy.io import loadmat
import random
from random import randint

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
    def __init__(self, architecture, num_skip_layers, skip_layers_coeff, enc_dec_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(CNN, self).__init__()

        self.num_skip_layers = num_skip_layers
        self.skip_layers_coeff = skip_layers_coeff
        self.enc_dec_out = enc_dec_out

        #################### Coding Function and Scene Parameters
        sourceExponent = 9
        ambientExponent = 6

        #### Parametrized Coding (Fourier Series)
        N = 10000
        self.N = N
        K = 3
        self.K = K
        order = 30 # The number of sinusoids to sum per function
        self.order = order

        # Initialize at Hamiltonian
        temp_alpha_mod = torch.zeros(K, order, device=device, dtype=dtype)
        temp_alpha_demod = torch.zeros(K, order, device=device, dtype=dtype)
        temp_phi_mod = torch.zeros(K, order, device=device, dtype=dtype)
        temp_phi_demod = torch.zeros(K, order, device=device, dtype=dtype)
        for i in range(self.order):
            temp_alpha_mod[:,i] = 2*6/((i+1)*math.pi) * np.sin((i+1)*math.pi/6) * torch.ones(K, device=device, dtype=dtype)
            temp_alpha_demod[:,i] = 2*1/((i+1)*math.pi) * np.sin((i+1)*math.pi/2) * torch.ones(K, device=device, dtype=dtype)
        for i in range(self.K):
            temp_phi_mod[i,:] = -1/12*math.pi * torch.ones(order, device=device, dtype=dtype)
            temp_phi_demod[i,:] = i*2/3*math.pi * torch.ones(order, device=device, dtype=dtype)
        self.alpha_mod = temp_alpha_mod.clone().detach().requires_grad_(True)
        self.alpha_demod = temp_alpha_demod.clone().detach().requires_grad_(True)
        self.phi_mod = temp_phi_mod.clone().detach().requires_grad_(True)
        self.phi_demod = temp_phi_demod.clone().detach().requires_grad_(True)

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
        
        #### Calculate current coding functions based on learned parameters
        ModFs_func = torch.zeros(self.N, self.K, device=device, dtype=dtype, requires_grad=False)
        DemodFs_func = torch.zeros(self.N, self.K, device=device, dtype=dtype, requires_grad=False)
        ModFs = torch.zeros(self.N, self.K, device=device, dtype=dtype, requires_grad=False)
        DemodFs = torch.zeros(self.N, self.K, device=device, dtype=dtype, requires_grad=False)
        p = torch.linspace(0, 2*math.pi, self.N, device=device)
        for k in range(0, self.K):
            for order in range(0, self.order):
                ModFs_func[:, k] += self.alpha_mod[k, order] * torch.cos((p+self.phi_mod[k, order])*(order+1))
                DemodFs_func[:, k] += self.alpha_demod[k, order] * torch.cos((p+self.phi_demod[k, order])*(order+1))
            
        # Normalize ModFs and DemodFs
        min_ModFs_func, _ = torch.min(ModFs_func, dim=0)
        ModFs = ModFs_func - min_ModFs_func # ModFs can't be lower than zero (negative light)
        min_DemodFs_func, _ = torch.min(DemodFs_func, dim=0)
        max_DemodFs_func, _ = torch.max(DemodFs_func, dim=0)
        DemodFs = (DemodFs_func - min_DemodFs_func) / (max_DemodFs_func - min_DemodFs_func) # DemodFs can only be 0->1

        #################### Simulation
        ## Set area under the curve of outgoing ModF to the totalEnergy
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
        BVals = BVals.permute(0,3,1,2) # Put channel dimension at right position

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
                self.layer_down2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU())
                self.layer_down3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU())

                self.layer_same1 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU())

                self.layer_up1 = nn.Sequential(
                    nn.BatchNorm2d(128),
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                    nn.ReLU())
                self.layer_up2 = nn.Sequential(
                    nn.BatchNorm2d(64),
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU())
                self.layer_up3 = nn.Sequential(
                    nn.BatchNorm2d(32),
                    nn.ConvTranspose2d(32, self.enc_dec_out, kernel_size=4, stride=2, padding=1),
                    nn.ReLU())

                if self.num_skip_layers == 1:
                    self.layer_skip_nonlinearity1 = nn.Sequential(
                        nn.Conv2d(self.K, self.skip_layers_coeff*16, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(self.skip_layers_coeff*16),
                        nn.ReLU())

                if self.num_skip_layers == 2:
                    self.layer_skip_nonlinearity1 = nn.Sequential(
                        nn.Conv2d(self.K, self.skip_layers_coeff*8, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(self.skip_layers_coeff*8),
                        nn.ReLU())
                    self.layer_skip_nonlinearity2 = nn.Sequential(
                        nn.Conv2d(self.skip_layers_coeff*8, self.skip_layers_coeff*16, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(self.skip_layers_coeff*16),
                        nn.ReLU())

                self.layer_combine = nn.Sequential(
                    nn.Conv2d(self.enc_dec_out + self.skip_layers_coeff*16,1, kernel_size=1, stride=1, padding=0))
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
                # Skip layer and combination with CNN
                if self.num_skip_layers == 1:
                    x_skip = self.layer_skip_nonlinearity1(BVals)
                if self.num_skip_layers == 2:
                    x_s1 = self.layer_skip_nonlinearity1(BVals)
                    x_skip = self.layer_skip_nonlinearity2(x_s1)
                x = self.layer_combine(torch.cat([x, x_skip], 1))
                return x


# Load data
#data = loadmat('patches_64.mat')
data = loadmat('patches_train_test_val_64.mat')
train = data['patches_train']
val = data['patches_val']

#val = torch.from_numpy(train[50000:51000,:,:])
train = torch.from_numpy(train[:50000,:,:])
val = torch.from_numpy(val)

train_gt_depths = train.float().to(device).requires_grad_(True)
val_gt_depths = val.float().to(device).requires_grad_(False)

train_gt_depths_mean = torch.mean(train_gt_depths)
train_gt_depths_std = torch.std(train_gt_depths)

train_normalized_gt_depths = (train_gt_depths-train_gt_depths_mean)/train_gt_depths_std
val_normalized_gt_depths = (val_gt_depths-train_gt_depths_mean)/train_gt_depths_std

print("DATA IMPORTED")

NUM_SKIP_LAYERS = [1, 2]
SKIP_LAYERS_COEFF = [1, 2]
ENC_DEC_OUT = [1, 3, 8, 16, 32]
LEARNING_RATE = [1e-4, 3e-4]
search = 1
file = open("hyperparam_results", "w")
for num_skip_layers in NUM_SKIP_LAYERS:
    for skip_layers_coeff in SKIP_LAYERS_COEFF:
        for enc_dec_out in ENC_DEC_OUT:
            for learning_rate in LEARNING_RATE:
                # Construct our model by instantiating the class defined above
                # Choose from: 'sequential', 'skip_connection'
                model = CNN('sequential', num_skip_layers, skip_layers_coeff, enc_dec_out)
                if use_gpu:
                    model.cuda()
                print("MODEL MADE")
                # Construct our loss function and an Optimizer. The call to model.parameters()
                # in the SGD constructor will contain the learnable parameters of the two
                # nn.Linear modules which are members of the model.
                criterion = torch.nn.MSELoss(reduction='mean')
                parameters = list(model.parameters())
                parameters.append(model.alpha_mod)
                parameters.append(model.alpha_demod)
                parameters.append(model.phi_mod)
                parameters.append(model.phi_demod)
                optimizer = optim.Adam(parameters, lr = learning_rate)


                with torch.autograd.detect_anomaly():
                    iteration = 0
                    increased = 0
                    patience = 5
                    train_batch_size = 64
                    val_every = 100
                    train_enumeration = torch.arange(train_gt_depths.shape[0])
                    train_enumeration = train_enumeration.tolist()
                    train_loss_history = []
                    val_loss_history = []

                    while increased <= patience:
                        train_ind = random.sample(train_enumeration, train_batch_size)
                        # Forward pass: Compute predicted y by passing x to the model
                        train_depths_pred = model(train_gt_depths[train_ind,:,:])
                        # Compute and print loss
                        train_loss = criterion(train_depths_pred, train_normalized_gt_depths[train_ind])
                        train_loss_history.append(train_loss.item())
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
                                val_depths_pred = model(val_gt_depths)
                                val_loss = criterion(val_depths_pred, val_normalized_gt_depths)
                                val_loss_history.append(val_loss.item())
                                val_depths_pred_unnorm = val_depths_pred*train_gt_depths_std+train_gt_depths_mean
                                val_MSE = criterion(val_depths_pred_unnorm, val_gt_depths)

                            print("Iteration: %d, Train Loss: %f, Val Loss: %f, Train MSE: %f, Val MSE: %f" %(iteration, train_loss.item(), val_loss.item(), train_MSE, val_MSE))
                            if iteration == 1 or val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_iteration = iteration
                                best_model = model
                                # torch.save(model,'./models/best_model_param_K1')
                                increased = 0
                            else:
                                increased = increased + 1
                print("**************DONE TRAINING*******************")
                file.write("**************DONE TRAINING*******************" + "\n")
                print("Number of Skip Layers:", num_skip_layers)
                file.write("Number of Skip Layers:" + str(num_skip_layers) + "\n")
                print("Number of Skip Layers Coefficient:", skip_layers_coeff)
                file.write("Number of Skip Layers Coefficient:" + str(skip_layers_coeff) + "\n")
                print("Encode Decode Output Num:", enc_dec_out)
                file.write("Encode Decode Output Num:" + str(enc_dec_out) + "\n")
                print("Learning Rate:", learning_rate)
                file.write("Learning Rate:" + str(learning_rate) + "\n")
                print("Best Validation Loss:", best_val_loss.item())
                file.write("Best Validation Loss:" + str(best_val_loss.item()) + "\n")
                print("Best Iteration:", best_iteration)
                file.write("Best Iteration:" + str(best_iteration) + "\n")
                print("**********************************************")
                file.write("**********************************************" + "\n")
                if search == 1:
                    GLOBAL_best_val_loss = best_val_loss
                    GLOBAL_num_skip_layers = num_skip_layers
                    GLOBAL_skip_layers_coeff = skip_layers_coeff
                    GLOBAL_enc_dec_out = enc_dec_out
                    GLOBAL_learning_rate = learning_rate
                search = search + 1
                if best_val_loss < GLOBAL_best_val_loss:
                    GLOBAL_best_val_loss = best_val_loss
                    GLOBAL_num_skip_layers = num_skip_layers
                    GLOBAL_skip_layers_coeff = skip_layers_coeff
                    GLOBAL_enc_dec_out = enc_dec_out
                    GLOBAL_learning_rate = learning_rate

print("********************************************************")
file.write("********************************************************" + "\n")
print("*********************SEARCH BEST************************")
file.write("*********************SEARCH BEST************************" + "\n")
print("********************************************************")
file.write("********************************************************" + "\n")
print("Number of Skip Layers:", GLOBAL_num_skip_layers)
file.write("Number of Skip Layers:" + str(GLOBAL_num_skip_layers) + "\n")
print("Number of Skip Layers Coefficient:", GLOBAL_num_skip_layers)
file.write("Number of Skip Layers Coefficient:" + str(GLOBAL_num_skip_layers) + "\n")
print("Encode Decode Output Num:", GLOBAL_enc_dec_out)
file.write("Encode Decode Output Num:" + str(GLOBAL_enc_dec_out) + "\n")
print("Learning Rate:", GLOBAL_learning_rate)
file.write("Learning Rate:" + str(GLOBAL_learning_rate) + "\n")
print("Best Validation Loss:", GLOBAL_best_val_loss.item())
file.write("Best Validation Loss:" + str(GLOBAL_best_val_loss.item()) + "\n")
print("********************************************************")
file.write("********************************************************" + "\n")
print("********************************************************")
file.write("********************************************************" + "\n")
print("********************************************************")
file.write("********************************************************" + "\n")
file.close()
