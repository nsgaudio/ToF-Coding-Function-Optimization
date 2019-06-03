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
    def __init__(self, architecture, K):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(CNN, self).__init__()

        #################### Coding Function and Scene Parameters
        sourceExponent = 9
        ambientExponent = 6

        #### Coding (Initialize at Hamiltonian)
        self.N = 10000
        self.K = K
        (ModFs_np,DemodFs_np) = CodingFunctions.GetCosCos(N = self.N, K=self.K)
        temp = torch.tensor(ModFs_np, device=device, dtype=dtype)
        self.ModFs = temp[:,:K].clone().detach().requires_grad_(True)
        temp = torch.tensor(DemodFs_np, device=device, dtype=dtype)
        self.DemodFs = temp[:,:K].clone().detach().requires_grad_(True)

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
        self.dt = self.tauMin/float(self.N)
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
        # Calculate correlation functions (NxK matrix) and normalize it (zero mean, unit variance)
        CorrFs = Utils.GetCorrelationFunctions(ModFs_scaled,self.DemodFs,device=device,dt=self.dt)
        NormCorrFs = (CorrFs.t() - torch.mean(CorrFs,1)) / torch.std(CorrFs,1)
        NormCorrFs = NormCorrFs.t()
        # Compute brightness values
        BVals = Utils.ComputeBrightnessVals(ModFs=ModFs_scaled, DemodFs=self.DemodFs, CorrFs=CorrFs, depths=gt_depths, \
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
                    nn.ConvTranspose2d(32, 8, kernel_size=4, stride=2, padding=1),
                    nn.ReLU())

                self.layer_skip_nonlinearity1 = nn.Sequential(
                    nn.Conv2d(self.K, 8, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(8),
                    nn.ReLU())
                self.layer_skip_nonlinearity2 = nn.Sequential(
                    nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(16),
                    nn.ReLU())


                self.layer_combine = nn.Sequential(
                    nn.Conv2d(24,1, kernel_size=1, stride=1, padding=0))
               
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
                x_skip = self.layer_skip_nonlinearity1(BVals)
                x_skip = self.layer_skip_nonlinearity2(x_skip)
                x = self.layer_combine(torch.cat([x, x_skip], 1))
                return x

 
# Load data
data = loadmat('patches_train_test_val_64.mat')
train = data['patches_train']
val = data['patches_val']
test = data['patches_test']

#val = torch.from_numpy(train[50000:51000,:,:])
#test = torch.from_numpy(train[70000:71000,:,:])
train = torch.from_numpy(train[:50000,:,:])
val = torch.from_numpy(val)
test = torch.from_numpy(test)

train_gt_depths = train.float().to(device).requires_grad_(True)
val_gt_depths = val.float().to(device).requires_grad_(False)
test_gt_depths = test.float().to(device).requires_grad_(False)

train_gt_depths_mean = torch.mean(train_gt_depths)
train_gt_depths_std = torch.std(train_gt_depths)

train_normalized_gt_depths = (train_gt_depths-train_gt_depths_mean)/train_gt_depths_std
val_normalized_gt_depths = (val_gt_depths-train_gt_depths_mean)/train_gt_depths_std
test_normalized_gt_depths = (test_gt_depths-train_gt_depths_mean)/train_gt_depths_std
print("DATA IMPORTED")

K_NUMBER = [1, 2]
for K in K_NUMBER:
    # Construct our model by instantiating the class defined above
    # Choose from: 'sequential', 'skip_connection'
    model = CNN('sequential', K)
    if use_gpu:
        model.cuda()
    print("MODEL MADE")
    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='mean')
    parameters = list(model.parameters())
    parameters.append(model.ModFs)
    parameters.append(model.DemodFs)
    optimizer = optim.Adam(parameters, lr = 3e-4)

    with torch.autograd.detect_anomaly():
        iteration = 0
        increased = 0
        patience = 100
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
                torch.cuda.empty_cache()

            if iteration == 1 or iteration%val_every == 0:
                with torch.no_grad():
                    val_depths_pred = model(val_gt_depths)
                    val_loss = criterion(val_depths_pred, val_normalized_gt_depths)
                    val_loss_history.append(val_loss.item())
                    val_depths_pred_unnorm = val_depths_pred*train_gt_depths_std+train_gt_depths_mean
                    val_MSE = criterion(val_depths_pred_unnorm, val_gt_depths)
                print("Iteration: %d, K: %d, Train Loss: %f, Val Loss: %f, Train MSE: %f, Val MSE: %f" %(iteration, K, train_loss.item(), val_loss.item(), train_MSE, val_MSE))
                if iteration == 1 or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_iteration = iteration
                    best_model = model
                    model_name = 'results/model_nn_K' + str(K) + '_patched_point'
                    torch.save(model, model_name)
                    increased = 0
                else:
                    increased = increased + 1

    results = "results/results_nn_K" + str(K) + "_patched_point"
    file = open(results, "w")
    print("DONE TRAINING")
    print("Best Validation Loss:", best_val_loss.item())
    file.write("Best Validation Loss:" + str(best_val_loss.item()) + "\n")
    print("Best Iteration:", best_iteration)
    file.write("Best Iteration:" + str(best_iteration) + "\n")

    # Plot training and validation loss over iterations
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(train_loss_history)), train_loss_history,  label='training loss')
    ax.plot(np.arange(0,len(val_loss_history)*val_every,val_every), val_loss_history, label='validation loss')
    ax.legend(loc='best')
    # plt.show(block=True)
    plot_name = 'results/loss_history_nn_K' + str(K) + '_patched_point.png'
    plt.savefig(plot_name)

    # Loss on test set
    with torch.no_grad():
        test_depths_pred = best_model(test_gt_depths)
        test_loss = criterion(test_depths_pred, test_normalized_gt_depths)
        test_depths_pred_unnorm = test_depths_pred*train_gt_depths_std+train_gt_depths_mean
        test_MSE = criterion(test_depths_pred_unnorm, test_gt_depths)
    print("Evaluate best model on test set:")
    print("Test Loss: %f, Test MSE: %f" %(test_loss.item(), test_MSE))
    file.write("Test Loss:" + str(test_loss.item()) + ", Test MSE:" + str(test_MSE) + "\n")

    # Stitch test patches back together and show two example depth maps from test set
    D = 64
    row_patch_num = 7
    col_patch_num = 9
    patch_num_scene = row_patch_num * col_patch_num
    test_depths_pred_unnorm_cpu = test_depths_pred_unnorm.cpu().numpy()
    test_gt_depths_cpu = test_gt_depths.cpu().numpy()
    num = np.floor(test_depths_pred_unnorm_cpu.shape[0] / patch_num_scene)
    n1 = randint(0, num-1)
    n2 = randint(0, num-1)
    im1 = np.zeros((row_patch_num*D, col_patch_num*D))
    im1_gt = np.zeros((row_patch_num*D, col_patch_num*D))
    im2 = np.zeros((row_patch_num*D, col_patch_num*D))
    im2_gt = np.zeros((row_patch_num*D, col_patch_num*D))
    ind1 = int(patch_num_scene * n1)
    ind2 = int(patch_num_scene * n2)
    for r in range(row_patch_num):
        for c in range(col_patch_num):
            im1[r*D:(r+1)*D, c*D:(c+1)*D] = np.squeeze(test_depths_pred_unnorm_cpu[ind1, :, :])
            im1_gt[r*D:(r+1)*D, c*D:(c+1)*D] = np.squeeze(test_gt_depths_cpu[ind1, :, :])
            im2[r*D:(r+1)*D, c*D:(c+1)*D] = np.squeeze(test_depths_pred_unnorm_cpu[ind2, :, :])
            im2_gt[r*D:(r+1)*D, c*D:(c+1)*D] = np.squeeze(test_gt_depths_cpu[ind2, :, :])
            ind1 = ind1 + 1
            ind2 = ind2 + 1
    im1_max = np.amax(im1_gt)
    im1_min = np.amin(im1_gt)
    im2_max = np.amax(im2_gt)
    im2_min = np.amin(im2_gt)

    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im1_gt)
    plt.set_cmap('jet')
    plt.clim(im1_min,im1_max)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(im1)
    plt.set_cmap('jet')
    plt.clim(im1_min,im1_max)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(im1_gt-im1)
    plt.set_cmap('bwr')
    plt.clim(-500,500)
    plt.colorbar()
    # plt.show(block=True)
    plot_name = 'results/depth_plot_nn_one_K' + str(K) + '_patched_point.png'
    plt.savefig(plot_name)

    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im2_gt)
    plt.set_cmap('jet')
    plt.clim(im2_min,im2_max)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(im2)
    plt.set_cmap('jet')
    plt.clim(im2_min,im2_max)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(im2_gt-im2)
    plt.set_cmap('bwr')
    plt.clim(-500,500)
    plt.colorbar()
    # plt.show(block=True)
    plot_name = 'results/depth_plot_nn_two_K' + str(K) + '_patched_point.png'
    plt.savefig(plot_name)


    # Save coding functions
    ModFs_scaled = Utils.ScaleMod(model.ModFs, device=device, tau=model.tauMin, pAveSource=model.pAveSourcePerPixel)

    UtilsPlot.PlotCodingScheme(model.ModFs,model.DemodFs,device)
    ModFs_np = model.ModFs.cpu().detach().numpy()
    DemodFs_np = model.DemodFs.cpu().detach().numpy()
    CorrFs = Utils.GetCorrelationFunctions(model.ModFs,model.DemodFs,device=device)
    CorrFs_np = CorrFs.cpu().detach().numpy()
    name = 'results/coding_functions_nn_K' + str(K) + '_patched_point.npz'
    np.savez(name, ModFs=ModFs_np, DemodFs=DemodFs_np, CorrFs=CorrFs_np)
    file.close()

