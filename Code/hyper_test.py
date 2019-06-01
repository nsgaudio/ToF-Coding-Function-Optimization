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

dtype = torch.float
device = torch.device("cpu")
use_gpu = True
if use_gpu:
    device = torch.device("cuda:0") # Uncomment this to run on GPU



NUM_SKIP_LAYERS = [1, 2]
## Coefficient in front of skip layer 1, 2, 3 (i.e. L1= K, L2=coeff*8, L3=coeff*16)
SKIP_LAYERS_COEFF = [1, 2]
ENC_DEC_OUT = [1, 3, 8, 16, 32]
LEARNING_RATE = [1e-4, 3e-4]
i = 1
for num_skip_layers in NUM_SKIP_LAYERS:
    for skip_layers_coeff in SKIP_LAYERS_COEFF:
        for enc_dec_out in ENC_DEC_OUT:
            for learning_rate in LEARNING_RATE:
                # print(num_skip_layers, skip_layers_coeff, enc_dec_out, learning_rate)
                print(i)
                i = i+1


