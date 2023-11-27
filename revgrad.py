import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import USPS
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

import config
from data import MNISTM, SVHN
from models import MNIST_MNISTM, MNIST_USPS, SVHN_MNIST
from utils import GrayscaleToRgb, PadSize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

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
    epochs = 10
    model_file = 'trained_models/source.pt'
    
    source = 'SVHN'
    target = 'MNIST'

    if target == 'MNIST-M':
        model = MNIST_MNISTM().to(device)
    elif target == 'USPS':
        model = MNIST_USPS().to(device)
    elif source == 'SVHN' and target == 'MNIST':
        model = SVHN_MNIST().to(device)
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

    else:
        raise ValueError(f'Invalid target dataset: {target}')

    # We're gonna make batches combining source and target data,
    # so we need to halve the batch size for each dataset

    half_batch = batch_size // 2
    
    if target == 'MNIST-M':
        source_dataset = MNIST(config.DATA_DIR/'mnist', train=True, download=True,
                          transform=Compose([GrayscaleToRgb(), ToTensor()]))
        target_dataset = MNISTM(train=False)
    elif target == 'USPS':
        source_dataset = MNIST(config.DATA_DIR/'mnist', train=True, download=True, transform=Compose([ToTensor()]))
        target_dataset = USPS(config.DATA_DIR/'usps', train=True, download=True, transform=Compose([ToTensor()]))
    elif source == 'SVHN' and target == 'MNIST':
        source_dataset = MNIST(config.DATA_DIR/'mnist', train=True, download=True, transform=Compose([PadSize(target_size=(32,32)), GrayscaleToRgb(), ToTensor()]))
        target_dataset = SVHN()
    else:
        raise ValueError(f'Invalid target dataset: {target}')

    source_loader = DataLoader(source_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)
    target_loader = DataLoader(target_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=1, pin_memory=True)

    optim = torch.optim.Adam(list(discriminator.parameters()) + list(model.parameters()))

    # for epoch in range(1, epochs+1):
    #     batches = zip(source_loader, target_loader)
    #     n_batches = min(len(source_loader), len(target_loader))

    #     total_domain_loss = total_label_accuracy = 0
    #     for (source_x, source_labels), (target_x, _) in tqdm(batches, leave=False, total=n_batches):
    #             # combined batch
    #             x = torch.cat([source_x, target_x])
    #             x = x.to(device)

    #             # true labels for domain classifier (0 for source, 1 for target)
    #             domain_y = torch.cat([torch.ones(source_x.shape[0]),
    #                                   torch.zeros(target_x.shape[0])])
    #             domain_y = domain_y.to(device)
    #             label_y = source_labels.to(device)

    #             features = feature_extractor(x).view(x.shape[0], -1)
    #             # predictions for domain classifier
    #             domain_preds = discriminator(features).squeeze()
    #             label_preds = clf(features[:source_x.shape[0]])
                
    #             domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
    #             label_loss = F.cross_entropy(label_preds, label_y)
    #             loss = domain_loss + label_loss

    #             optim.zero_grad()
    #             loss.backward()
    #             optim.step()

    #             total_domain_loss += domain_loss.item()
    #             total_label_accuracy += (label_preds.max(1)[1] == label_y).float().mean().item()

    #     mean_loss = total_domain_loss / n_batches
    #     mean_accuracy = total_label_accuracy / n_batches
    #     tqdm.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}, '
    #                f'source_accuracy={mean_accuracy:.4f}')

    #     torch.save(model.state_dict(), 'trained_models/revgrad.pt')

    for epoch in range(1, epochs+1):
        batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))

        total_domain_loss = total_label_accuracy = 0
        for (source_x, source_labels), (target_x, _) in tqdm(batches, leave=False, total=n_batches):
            # Move the source and target data to the device
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
            
            # Forward pass for target data
            target_features = feature_extractor(target_x).view(target_x.shape[0], -1)
            target_domain_preds = discriminator(target_features).squeeze()
            target_label_preds = clf(target_features)

            # Compute the losses
            source_domain_loss = F.binary_cross_entropy_with_logits(source_domain_preds, source_domain_y)
            target_domain_loss = F.binary_cross_entropy_with_logits(target_domain_preds, target_domain_y)
            label_loss = F.cross_entropy(source_label_preds, label_y)
            loss = source_domain_loss + target_domain_loss + label_loss

            # Backward pass and optimization
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
