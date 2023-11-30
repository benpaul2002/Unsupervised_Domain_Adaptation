import numpy as np
import torch
from torch import nn, optim
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

class Office31_Dataset(Dataset):
    def __init__(self, domain='Amazon', image_size=(224, 224)):
        super(Office31_Dataset, self).__init__()
        self.office31 = ImageFolder('/home/ben/Documents/Year_4/Sem_7/SMAI/Project/Code/Try3/data/office31/'+domain, transform=Compose([Resize(image_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    def __getitem__(self, index: int):
        img, target = self.office31[index]
        return img, target

    def __len__(self):
        return len(self.office31)
    
class Office(nn.Module):
    def __init__(self, num_classes=31):
        super(Office, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 31
batch_size = 64

def create_train_val_loader(batch_size, dataset):
  shuffled_indices = np.random.permutation(len(dataset))
  train_idx = shuffled_indices[:int(0.8*len(dataset))]       
  val_idx = shuffled_indices[int(0.8*len(dataset)):]

  train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(train_idx),
                            num_workers=1, pin_memory=True)
  val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, sampler=SubsetRandomSampler(val_idx),
                          num_workers=1, pin_memory=True)
  
  return train_loader, val_loader

source_dataset = Office31_Dataset(domain="Amazon")
train_loader, val_loader = create_train_val_loader(batch_size, source_dataset)

model = Office(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples

print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(val_loader, model)*100:.2f}")
