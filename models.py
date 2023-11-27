from torch import nn

class MNIST_MNISTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Dropout2d(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits

class MNIST_USPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),        # 28x28 -> 24x24x10 & 16x16 -> 12x12x10
            nn.AdaptiveAvgPool2d((12, 12)),           # 24x24x10 -> 12x12x10 & 12x12x10 -> 4x4x10
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),       # 12x12x10 -> 8x8x20
            nn.MaxPool2d(2),                        # 8x8x20 -> 4x4x20
            nn.Dropout2d(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits
    
class SVHN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),        # 32x32 -> 28x28x64             & 28x28 -> 24x24x64
            nn.ReLU(),                              # 28x28x64 -> 28x28x64         & 24x24x64 -> 24x24x64    
            nn.Conv2d(64, 64, kernel_size=5),       # 28x28x64 -> 24x24x64          & 24x24x64 -> 20x20x64
            nn.ReLU(),                              # 24x24x64 -> 24x24x64          & 20x20x64 -> 20x20x64
            nn.MaxPool2d(3, 2),                     # 24x24x64 -> 11x11x64          & 20x20x64 -> 9x9x64
            nn.Conv2d(64, 128, kernel_size=5),      # 11x11x64 -> 7x7x128           & 9x9x64 -> 5x5x128
            nn.ReLU(),                              # 7x7x128 -> 7x7x128            & 5x5x128 -> 5x5x128
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(6272, 3072),
            nn.ReLU(),
            nn.Linear(3072, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits

