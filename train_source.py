import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

import config
from models import MNIST_MNISTM, MNIST_USPS
from utils import GrayscaleToRgb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_dataloaders(batch_size, target):
    '''
    Create dataloaders for MNIST train and validation sets.
    train: 48,000 samples -> 80% of the dataset
    val: 12,000 samples -> 20% of the dataset
    Since this is MNIST -> MNIST-M and our neural net expects 3 channels, 
    we use GrayscaleToRgb transform.
    '''
    if target == 'MNIST-M':
        dataset = MNIST(config.DATA_DIR/'mnist', train=True, download=True,
                        transform=Compose([GrayscaleToRgb(), ToTensor()]))
    elif target == 'USPS':
        dataset = MNIST(config.DATA_DIR/'mnist', train=True, download=True,
                        transform=Compose([ToTensor()]))
    else:
        raise ValueError(f'Invalid target dataset: {target}')
    
    shuffled_indices = np.random.permutation(len(dataset))
    train_idx = shuffled_indices[:int(0.8*len(dataset))]       
    val_idx = shuffled_indices[int(0.8*len(dataset)):]

    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=1, pin_memory=True)
    return train_loader, val_loader

def do_epoch(model, dataloader, criterion, optim=None):
    '''
    Train or evaluate model for one epoch.
    Iterate over all batches in dataloader.
    criterion is loss function defined in main
    during training, optim is not None, so we do backpropagation
    y_pred is 64x10 tensor
    y_pred.max(1) returns 2 tensors of size 64
    y_pred.max(1)[0] is the max value for each of the 64 samples -> represents the confidence of the prediction
    y_pred.max(1)[1] is the index of the max value for each of the 64 samples -> represents the predicted class
    compare y_pred.max(1)[1] with y_true -> returns a tensor of size 64 with 1s and 0s
    .float() converts 1s and 0s to 1.0s and 0.0s
    .mean() computes the mean of the tensor -> accuracy
    '''
    total_loss = 0
    total_accuracy = 0
    for x, y_true in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y_true)

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy

def main():
    batch_size = 64
    target = 'USPS'

    train_loader, val_loader = create_dataloaders(batch_size, target)

    if target == 'MNIST-M':
        model = MNIST_MNISTM().to(device)
        epochs = 30
    elif target == 'USPS':
        model = MNIST_USPS().to(device)
        epochs = 50
    else:
        raise ValueError(f'Invalid target dataset: {target}')
    
    optim = torch.optim.Adam(model.parameters())
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(1, epochs+1):
        model.train()
        train_loss, train_accuracy = do_epoch(model, train_loader, criterion, optim=optim)

        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = do_epoch(model, val_loader, criterion, optim=None)

        tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                   f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            print('Saving model...')
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'trained_models/source.pt')

        lr_schedule.step(val_loss)

main()
