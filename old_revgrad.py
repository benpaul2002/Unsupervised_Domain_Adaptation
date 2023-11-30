import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import USPS
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from data import MNISTM, SVHN, Office31
from models import MNIST_MNISTM, MNIST_USPS, SVHN_MNIST, Office
from utils import GrayscaleToRgb, PadSize
from pathlib import Path

DATA_DIR = Path('/home/ben/Documents/Year_4/Sem_7/SMAI/Project/Code/Try1/data')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def main():
    batch_size = 64
    epochs = 30
    model_file = 'trained_models/source.pt'
    
    source = 'MNIST_MNIST-M'
    target = 'SVHN'

    if target == 'MNIST-M':
        model = MNIST_MNISTM().to(device)
    elif target == 'USPS':
        model = MNIST_USPS().to(device)
    elif source == 'SVHN' and target == 'MNIST':
        model = SVHN_MNIST().to(device)
    elif source == 'amazon' and target == 'webcam':
        model = Office(domain=source).to(device)
    elif source == 'webcam' and target == 'dslr':
        model = Office(domain=source).to(device)
    elif source == 'MNIST_MNIST-M' and target == 'SVHN':
        model = MNIST_MNISTM().to(device)
    else:
        raise ValueError(f'Invalid target dataset: {target}')
    
    model.load_state_dict(torch.load(model_file))
    feature_extractor = model.feature_extractor
    clf = model.classifier

    if target == 'MNIST-M':
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        ).to(device)

    elif target == 'USPS':
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        ).to(device)

    elif source == 'SVHN' and target == 'MNIST':
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(6272, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        ).to(device)

    elif source == 'amazon' and target == 'webcam':
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(9216, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        ).to(device)

    elif source == 'webcam' and target == 'dslr':
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(9216, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        ).to(device)

    elif source == 'MNIST_MNIST-M' and target == 'SVHN':
        discriminator = nn.Sequential(
            GradientReversal(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        ).to(device)

    else:
        raise ValueError(f'Invalid target dataset: {target}')

    # We're gonna make batches combining source and target data,
    # so we halve the batch size for each dataset

    half_batch = batch_size // 2

    source_dataset1 = None
    source_dataset2 = None
    
    if target == 'MNIST-M':
        source_dataset = MNIST(DATA_DIR/'mnist', train=True, download=True,
                          transform=Compose([GrayscaleToRgb(), ToTensor()]))
        target_dataset = MNISTM(train=False)
    elif target == 'USPS':
        source_dataset = MNIST(DATA_DIR/'mnist', train=True, download=True, transform=Compose([ToTensor()]))
        target_dataset = USPS(DATA_DIR/'usps', train=True, download=True, transform=Compose([ToTensor()]))
    elif source == 'SVHN' and target == 'MNIST':
        source_dataset = MNIST(DATA_DIR/'mnist', train=True, download=True, transform=Compose([PadSize(target_size=(32,32)), GrayscaleToRgb(), ToTensor()]))
        target_dataset = SVHN()
    elif source == 'amazon' and target == 'webcam':
        source_dataset = Office31(domain=source)
        target_dataset = Office31(domain=target)
    elif source == 'webcam' and target == 'dslr':
        source_dataset = Office31(domain=source)
        target_dataset = Office31(domain=target)
    elif source == 'MNIST_MNIST-M' and target == 'SVHN':
        source_dataset1 = MNIST(DATA_DIR/'mnist', train=True, download=True,
                          transform=Compose([PadSize(target_size=(32,32)), GrayscaleToRgb(), ToTensor()]))
        source_dataset2 = MNISTM(train=False)
        target_dataset = SVHN()
    else:
        raise ValueError(f'Invalid target dataset: {target}')


    if source_dataset1 is not None:
        source_loader1 = DataLoader(source_dataset1, batch_size=half_batch,
                                 shuffle=True, num_workers=1, pin_memory=True)
        source_loader2 = DataLoader(source_dataset2, batch_size=half_batch,
                                    shuffle=True, num_workers=1, pin_memory=True)
        source_loader = zip(source_loader1, source_loader2)
        
    else:
        source_loader = DataLoader(source_dataset, batch_size=half_batch,
                                shuffle=True, num_workers=1, pin_memory=True)
        
    target_loader = DataLoader(target_dataset, batch_size=half_batch,
                            shuffle=True, num_workers=1, pin_memory=True)

    optim = torch.optim.Adam(list(discriminator.parameters()) + list(model.parameters()))

    for epoch in range(1, epochs+1):
        batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))

        total_domain_loss = total_label_accuracy = 0
        for (source_x, source_labels), (target_x, _) in tqdm(batches, leave=False, total=n_batches):
            source_x = source_x.to(device)
            target_x = target_x.to(device)

            # True labels for domain classifier (0 for source, 1 for target)
            source_domain_y = torch.ones(source_x.shape[0]).to(device)
            target_domain_y = torch.zeros(target_x.shape[0]).to(device)
            label_y = source_labels.to(device)

            # Forward pass for source data
            source_features = feature_extractor(source_x).view(source_x.shape[0], -1)
            source_domain_preds = discriminator(source_features).squeeze()
            source_label_preds = clf(source_features)
            
            target_features = feature_extractor(target_x).view(target_x.shape[0], -1)
            target_domain_preds = discriminator(target_features).squeeze()
            target_label_preds = clf(target_features)

            source_domain_loss = F.binary_cross_entropy_with_logits(source_domain_preds, source_domain_y)
            target_domain_loss = F.binary_cross_entropy_with_logits(target_domain_preds, target_domain_y)
            label_loss = F.cross_entropy(source_label_preds, label_y)
            loss = source_domain_loss + target_domain_loss + label_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_domain_loss += source_domain_loss.item() + target_domain_loss.item()
            total_label_accuracy += (source_label_preds.max(1)[1] == label_y).float().mean().item()

        mean_loss = total_domain_loss / n_batches
        mean_accuracy = total_label_accuracy / n_batches
        tqdm.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}, '
                f'source_accuracy={mean_accuracy:.4f}')

    torch.save(model.state_dict(), 'trained_models/revgrad.pt')

main()
