# This file contains the implementation of the gradient reversal layer.
# It also has the function to get the domain discriminator.

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from pathlib import Path

DATA_DIR = Path("/home/ben/Documents/Year_4/Sem_7/SMAI/Project/Code/Try3/data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradientReversalFunction(Function):
    '''
        As described in the paper, we do nothing during forward pass.
        During backward pass, the grad_output is reversed (by multiplying
        by -1) and then scaled by lambda.
    '''
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    '''
        Custom PyTorch layer that implements gradient reversal.
    '''
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
def get_discriminator(source, target):
    '''
        Function to get appropriate discriminator based on source and target dataset.
        The input here is the output of the feature extractor.
    '''

    if source == "MNIST" and target == "MNIST-M":
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        ).to(device)

    elif source == "MNIST" and target == "USPS":
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        ).to(device)

    elif source == "MNIST" and target == "SVHN":
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(6272, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        ).to(device)

    elif source == "SVHN" and target == "MNIST":
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(6272, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        ).to(device)

    elif source == "Amazon" and target == "Webcam":
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(12500, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        ).to(device)

    elif source == "Webcam" and target == "DSLR":
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(12500, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        ).to(device)

    elif source == "DSLR" and target == "Amazon":
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(12500, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        ).to(device)

    elif source == "Quickdraw" and target == "Clipart":
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(9216, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        ).to(device)

    elif source == "Sketch" and target == "Painting":
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(9216, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        ).to(device)

    elif source == "Painting" and target == "Sketch":
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(9216, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        ).to(device)

    elif source == "MNIST_MNIST-M" and target == "SVHN":
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        ).to(device)

    else:
        raise ValueError(f"Invalid target dataset: {target}")
    
    return discriminator