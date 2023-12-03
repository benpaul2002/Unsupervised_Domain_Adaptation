from torch import nn
import torchvision.models as models

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
    
# class MNIST(nn.Module):
    
#     def __init__(self):
#         super().__init__()
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=5),  # 28x28 -> 24x24x32
#             nn.MaxPool2d(2, 2),             # 24x24x32 -> 12x12x32
#             nn.ReLU(),                      # 12x12x32 -> 12x12x32
#             nn.Conv2d(32, 48, kernel_size=5), # 12x12x32 -> 8x8x48
#             nn.MaxPool2d(2),                 # 8x8x48 -> 4x4x48
#             nn.Dropout2d(),                   # 4x4x48 -> 4x4x48
#         ) 
#         self.classifier = nn.Sequential(
#             nn.Linear(768, 100),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(100, 100),
#             nn.ReLU(),
#             nn.Linear(100, 10),
#         )

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         features = features.view(x.shape[0], -1)
#         logits = self.classifier(features)
#         return logits
        

          
    
    
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
    

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
        #     nn.BatchNorm2d(96),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size = 3, stride = 2))
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size = 3, stride = 2))
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU())
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU())
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size = 3, stride = 2))
        #feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        # self.fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(9216, 4096),
        #     nn.ReLU())
        # self.fc1 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU())
        # self.fc2= nn.Sequential(
        #     nn.Linear(4096, num_classes))
        #classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
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


