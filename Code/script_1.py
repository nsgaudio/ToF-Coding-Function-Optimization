import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn as nn            # containing various building blocks for your neural networks
import torch.optim as optim      # implementing various optimization algorithms
import torch.nn.functional as F  # a lower level (compared to torch.nn) interface

generated-members=torch.*

# Using this tutorial:
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

print("torch version:", torch.__version__)

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
        self.x = x

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.x

# Create random Tensors to hold inputs and outputs
x = torch.randn(10, 10, device=device, dtype=dtype, requires_grad=True)
x_init = x.clone()
y = torch.randn(10, 10, device=device, dtype=dtype, requires_grad=True)

# Construct our model by instantiating the class defined above
model = Pixelwise()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.SGD([x], lr=1e-4)
optimizer = optim.Adam([x], lr = 0.0001)
# optimizer = optim.Adam([x], lr = 0.0001, momentum=0.9)

for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    x_pred = model(x)

    # Compute and print loss
    loss = criterion(x_pred, y)
    # print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print("Initial x:", x_init)
print("Final x:", x_pred)
print("Difference:", x_pred - x_init)
