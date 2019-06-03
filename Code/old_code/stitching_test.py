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
from random import randint

# Local imports
import CodingFunctions
import Utils
import UtilsPlot
import Decoding

data = loadmat('patches_train_test_val_64.mat')
#data = loadmat('patches_448_576.mat')
train = data['patches_train']
val = data['patches_val']
test = data['patches_test'] 

train = torch.from_numpy(train)
val = torch.from_numpy(val)
test = torch.from_numpy(test)

D = 64
row_patch_num = 7
col_patch_num = 9
patch_num_scene = row_patch_num * col_patch_num

device = torch.device("cpu")

train_gt_depths = train.float().to(device).requires_grad_(True)
val_gt_depths = val.float().to(device).requires_grad_(False)
test_gt_depths = test.float().to(device).requires_grad_(False)

train_gt_depths_mean = torch.mean(train_gt_depths)
train_gt_depths_std = torch.std(train_gt_depths)

train_normalized_gt_depths = (train_gt_depths-train_gt_depths_mean)/train_gt_depths_std
val_normalized_gt_depths = (val_gt_depths-train_gt_depths_mean)/train_gt_depths_std
test_normalized_gt_depths = (test_gt_depths-train_gt_depths_mean)/train_gt_depths_std


test_depths_pred_unnorm = test_gt_depths.cpu().numpy()
test_gt_depths = test_gt_depths.cpu().numpy()
# Show two example depth maps from test set
num = np.floor(test_depths_pred_unnorm.shape[0] / patch_num_scene)
n1 = randint(0, num-1)
im1 = np.zeros((row_patch_num*D, col_patch_num*D))
im1_gt = np.zeros((row_patch_num*D, col_patch_num*D))
ind =  int(patch_num_scene * n1)
for r in range(row_patch_num):
    for c in range(col_patch_num):
        im1[r*D:(r+1)*D, c*D:(c+1)*D] = np.squeeze(test_depths_pred_unnorm[ind, :, :])
        im1_gt[r*D:(r+1)*D, c*D:(c+1)*D] = np.squeeze(test_depths_pred_unnorm[ind, :, :])
        ind = ind + 1

# im1 = test_depths_pred_unnorm[n1,:,:].cpu().numpy()
# im1_gt = test_gt_depths[n1,:,:].cpu().numpy()
im1_max = np.amax(im1_gt)
im1_min = np.amin(im1_gt)
# n2 = randint(0, num-1)
# im2 = test_depths_pred_unnorm[n2,:,:].cpu().numpy()
# im2_gt = test_gt_depths[n2,:,:].cpu().numpy()
# im2_max = np.amax(im2_gt)
# im2_min = np.amin(im2_gt)

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
plt.show(block=True)
