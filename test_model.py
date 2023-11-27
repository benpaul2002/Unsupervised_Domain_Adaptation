import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, USPS
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from data import MNISTM
from models import MNIST_MNISTM, MNIST_USPS

from utils import GrayscaleToRgb, PadSize

import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    batch_size = 256
    model_file = 'trained_models/revgrad.pt'

    source = 'MNIST'
    target = 'SVHN'

    if target == 'MNIST-M':
        dataset = MNISTM(train=False)
    elif target == 'USPS':
        dataset = USPS(config.DATA_DIR/'usps', train=False, download=True,
                       transform=Compose([ToTensor()]))
    elif source == 'SVHN' and target == 'MNIST':
        dataset = MNIST(config.DATA_DIR/'mnist', train=False, download=True,
                        transform=Compose([PadSize(target_size=(32,32)), GrayscaleToRgb(), ToTensor()]))
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=1, pin_memory=True)

    if target == 'MNIST-M':
        model = MNIST_MNISTM().to(device)
    elif target == 'USPS':
        model = MNIST_USPS().to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    total_accuracy = 0
    with torch.no_grad():
        for x, y_true in tqdm(dataloader, leave=False):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    
    mean_accuracy = total_accuracy / len(dataloader)
    print(f'Accuracy on target data: {mean_accuracy:.4f}')

main()
