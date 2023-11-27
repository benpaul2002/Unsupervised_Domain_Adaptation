import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize
import os
from PIL import Image
from pathlib import Path

DATA_DIR = Path('/home/ben/Documents/Year_4/Sem_7/SMAI/Project/Code/Try1/data')

class BSDS500(Dataset):
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

class MNISTM(Dataset):

    def __init__(self, train=True):
        super(MNISTM, self).__init__()
        self.mnist = datasets.MNIST(DATA_DIR / 'mnist', train=train,
                                    download=True)
        self.bsds = BSDS500()
        self.rng = np.random.RandomState(42)

    def __getitem__(self, i):
        digit, label = self.mnist[i]
        digit = ToTensor()(digit)
        bsds_image = self._random_bsds_image()
        patch = self._random_patch(bsds_image)
        patch = patch.float() / 255
        blend = torch.abs(patch - digit)
        return blend, label

    def _random_patch(self, image, size=(28, 28)):
        _, im_height, im_width = image.shape
        x = self.rng.randint(0, im_width-size[1])
        y = self.rng.randint(0, im_height-size[0])
        return image[:, y:y+size[0], x:x+size[1]]

    def _random_bsds_image(self):
        i = self.rng.choice(len(self.bsds))
        return self.bsds[i]

    def __len__(self):
        return len(self.mnist)
    
class SVHN(Dataset):
    def __init__(self):
        super(SVHN, self).__init__()
        self.svhn = datasets.SVHN(DATA_DIR / 'svhn', split='train',
                                   download=True, transform=ToTensor())
        self.bsds = BSDS500()
        self.rng = np.random.RandomState(42)

    def __getitem__(self, index: int):
        img, target = self.svhn[index]
        return img, target

    def __len__(self):
        return len(self.svhn)

# class Office31(Dataset):
#     def __init__(self, domain='amazon'):
#         super(Office31, self).__init__()
#         self.office31 = ImageFolder(DATA_DIR/'office31'/domain, transform=ToTensor())
#         self.rng = np.random.RandomState(42)

#     def __getitem__(self, index: int):
#         img, target = self.office31[index]
#         return img, target

#     def __len__(self):
#         return len(self.office31)

class Office31(Dataset):
    def __init__(self, domain='amazon', image_size=(128, 128)):
        super(Office31, self).__init__()
        self.office31 = ImageFolder(DATA_DIR/'office31'/domain, transform=ToTensor())
        self.resize_transform = Resize(image_size)

    def __getitem__(self, index: int):
        img, target = self.office31[index]
        img = self.resize_transform(img)
        return img, target

    def __len__(self):
        return len(self.office31)
