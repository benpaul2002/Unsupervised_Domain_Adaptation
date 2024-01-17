from torch import nn
import torchvision.models as models

class MNIST_MNISTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),        # 28x28 -> 24x24x10
            nn.MaxPool2d(2),                    # 24x24x10 -> 12x12x10
            nn.ReLU(),                        # 12x12x10 -> 12x12x10
            nn.Conv2d(10, 20, kernel_size=5),    # 12x12x10 -> 8x8x20
            nn.MaxPool2d(2),                # 8x8x20 -> 4x4x20
            nn.Dropout2d(),                   # 4x4x20 -> 4x4x20
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
    
class MNIST_MNISTM_SVHN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),        # 32 x 32 -> 28x28x10
            nn.MaxPool2d(2),                        # 28x28x10 -> 14x14x10
            nn.ReLU(),                              # 14x14x10 -> 14x14x10
            nn.Conv2d(10, 20, kernel_size=5),       # 14x14x10 -> 10x10x20
            nn.MaxPool2d(2),                        # 10x10x20 -> 5x5x20
            nn.Dropout2d(),                         # 5x5x20 -> 5x5x20
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(500, 50),
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
    
class DomainNet(nn.Module):
    def __init__(self, num_classes=31):
        super(DomainNet, self).__init__()
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

class Office(nn.Module):
    def __init__(self):
        super(Office, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),        # Assuming input images are 224x224 pixels
            nn.AdaptiveAvgPool2d((55, 55)),         # Downsample to 55x55
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2),                        # Downsample to 25x25
            nn.Dropout2d(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(20*25*25, 50),                # Flatten the tensor
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 31),                      # Assuming there are 31 classes in the Office31 dataset
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits

# class Office(nn.Module):
#     def __init__(self, num_classes=31):
#         super(Office, self).__init__()
#         alexnet = models.alexnet(pretrained=True)
#         self.feature_extractor = alexnet.features
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         features = features.view(x.shape[0], -1)
#         logits = self.classifier(features)
#         return logits

class ImageNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ImageNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits
