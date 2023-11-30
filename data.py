import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from torchvision.datasets import MNIST, USPS, SVHN
import os
from PIL import Image
from pathlib import Path
from utils import GrayscaleToRgb, PadSize

DATA_DIR = Path('/home/ben/Documents/Year_4/Sem_7/SMAI/Project/Code/Try3/data')

class MNIST_Dataset(Dataset):
    def __init__(self, grayscaleToRgb=False, padSize=False, image_size=(32, 32)):
        super(MNIST_Dataset, self).__init__()
        function_list = []
        if padSize:
            function_list.append(PadSize(target_size=image_size))
        if grayscaleToRgb:
            function_list.append(GrayscaleToRgb())
        function_list.append(ToTensor())
        self.mnist = MNIST(root=DATA_DIR/'mnist', train=True, download=True, transform=Compose(function_list))
        # self.resize_transform = Resize(image_size)

    def __getitem__(self, index: int):
        img, target = self.mnist[index]
        return img, target

    def __len__(self):
        return len(self.mnist)
    
class USPS_Dataset(Dataset):
    def __init__(self, padSize=False, image_size=(28, 28)):
        super(USPS_Dataset, self).__init__()
        function_list = []
        if padSize:
            function_list.append(PadSize(target_size=image_size))
        function_list.append(ToTensor())
        self.usps = USPS(root=DATA_DIR/'usps', train=True, download=True, transform=Compose(function_list))
        # self.resize_transform = Resize(image_size)

    def __getitem__(self, index: int):
        img, target = self.usps[index]
        # img = self.resize_transform(img)
        return img, target

    def __len__(self):
        return len(self.usps)

class BSDS500_Dataset(Dataset):
    def __init__(self):
        image_folder = DATA_DIR / 'BSR/BSDS500/data/images'
        print(image_folder)
        self.image_files = list(map(str, image_folder.glob('*/*.jpg')))

    def __getitem__(self, i):
        image = cv2.imread(self.image_files[i], cv2.IMREAD_COLOR)
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return tensor

    def __len__(self):
        return len(self.image_files)

# class MNISTM_Dataset(Dataset):
#     def __init__(self, train=True):
#         super(MNISTM_Dataset, self).__init__()
#         self.mnist = datasets.MNIST(DATA_DIR / 'mnist', train=train, download=True)
#         self.bsds = BSDS500_Dataset()
#         self.rng = np.random.RandomState(42)

#     def __getitem__(self, i):
#         digit, label = self.mnist[i]
#         digit = ToTensor()(digit)
#         bsds_image = self._random_bsds_image()
#         patch = self._random_patch(bsds_image)
#         patch = patch.float() / 255
#         blend = torch.abs(patch - digit)
#         return blend, label

#     def _random_patch(self, image, size=(28, 28)):
#         _, im_height, im_width = image.shape
#         x = self.rng.randint(0, im_width-size[1])
#         y = self.rng.randint(0, im_height-size[0])
#         return image[:, y:y+size[0], x:x+size[1]]

#     def _random_bsds_image(self):
#         i = self.rng.choice(len(self.bsds))
#         return self.bsds[i]

#     def __len__(self):
#         return len(self.mnist)

from torchvision import transforms

class MNISTM_Dataset(Dataset):
    def __init__(self, train=True, target_size=(28, 28)):
        super(MNISTM_Dataset, self).__init__()
        self.mnist = datasets.MNIST(DATA_DIR / 'mnist', train=train, download=True)
        self.bsds = BSDS500_Dataset()
        self.rng = np.random.RandomState(42)
        self.target_size = target_size

    def __getitem__(self, i):
        digit, label = self.mnist[i]
        digit = self.resize_image(digit, self.target_size)
        bsds_image = self._random_bsds_image()
        patch = self._random_patch(bsds_image)
        patch = patch.float() / 255
        blend = torch.abs(patch - digit)
        return blend, label

    def resize_image(self, image, size):
        transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        return transform(image)

    def _random_patch(self, image):
        _, im_height, im_width = image.shape
        x = self.rng.randint(0, im_width-self.target_size[1])
        y = self.rng.randint(0, im_height-self.target_size[0])
        return image[:, y:y+self.target_size[0], x:x+self.target_size[1]]

    def _random_bsds_image(self):
        i = self.rng.choice(len(self.bsds))
        return self.bsds[i]

    def __len__(self):
        return len(self.mnist)

    
class SVHN_Dataset(Dataset):
    def __init__(self):
        super(SVHN_Dataset, self).__init__()
        self.svhn = SVHN(DATA_DIR / 'svhn', split='train',
                                   download=True, transform=ToTensor())

    def __getitem__(self, index: int):
        img, target = self.svhn[index]
        return img, target

    def __len__(self):
        return len(self.svhn)

class Office31_Dataset(Dataset):
    def __init__(self, domain='Amazon', image_size=(227, 227)):
        super(Office31_Dataset, self).__init__()
        self.office31 = ImageFolder(DATA_DIR/'office31'/domain, transform=Compose([Resize(image_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    def __getitem__(self, index: int):
        img, target = self.office31[index]
        return img, target

    def __len__(self):
        return len(self.office31)

# class Office31_Dataset(Dataset):
#     def __init__(self, domain='Amazon', image_size=(227, 227)):
#         super(Office31_Dataset, self).__init__()
        
#         transform = Compose([
#             Resize(image_size),
#             RandomHorizontalFlip(),  # Add random horizontal flips for data augmentation
#             RandomVerticalFlip(),    # Add random vertical flips for additional augmentation
#             RandomRotation(degrees=15),  # Add random rotations for more variety
#             ToTensor(),
#             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#         self.office31 = ImageFolder(DATA_DIR/'office31'/domain, transform=transform)

#     def __getitem__(self, index: int):
#         try:
#             img, target = self.office31[index]
#         except Exception as e:
#             print(f"Error loading image at index {index}: {str(e)}")
#             return self.__getitem__((index + 1) % len(self.office31))  # Skip problematic image and try the next one

#         return img, target

#     def __len__(self):
#         return len(self.office31)


class ImageNet_Dataset(Dataset):
    def _init_(self, train=True):
        super(ImageNet_Dataset, self)._init_()
        self.imagenet = datasets.ImageNet(DATA_DIR / 'imagenet', split='train' if train else 'test',
                                    download=True)
        
    def _getitem_(self, i):
        digit, label = self.imagenet[i]
        digit = ToTensor()(digit)
        
        return digit, label

    def _len_(self):
        return len(self.imagenet)
    
class Caltech_Dataset(Dataset):
    def __init__(self, transform=None):
        self.root_dir = DATA_DIR / 'caltech'
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(class_path, filename)
                        images.append((image_path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

class DomainNet_Dataset(Dataset):
    def __init__(self, domain='Clipart', image_size=(224, 224)):
        super(DomainNet_Dataset, self).__init__()
        self.domain_net = ImageFolder(DATA_DIR/'domain_net'/domain, transform=Compose([Resize(image_size), ToTensor()]))

    def __getitem__(self, index: int):
        img, target = self.domain_net[index]
        return img, target

    def __len__(self):
        return len(self.domain_net)
    

class ConcatenatedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - len(self.dataset1)]
