import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from pathlib import Path

DATA_DIR = Path("/home/ben/Documents/Year_4/Sem_7/SMAI/Project/Code/Try3/data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradientReversalFunction(Function):
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
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
def get_discriminator(source, target):
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
            nn.Linear(9216, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        ).to(device)

    elif source == "Webcam" and target == "DSLR":
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
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        ).to(device)

    else:
        raise ValueError(f"Invalid target dataset: {target}")
    
    return discriminator