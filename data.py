import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
from PIL import Image

import config

class BSDS500(Dataset):

    def __init__(self):
        image_folder = config.DATA_DIR / 'BSR/BSDS500/data/images'
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
        self.mnist = datasets.MNIST(config.DATA_DIR / 'mnist', train=train,
                                    download=True)
        self.bsds = BSDS500()
        # Fix RNG so the same images are used for blending
        self.rng = np.random.RandomState(42)

    def __getitem__(self, i):
        digit, label = self.mnist[i]
        digit = transforms.ToTensor()(digit)
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
        self.svhn = datasets.SVHN(config.DATA_DIR / 'svhn', split='train',
                                   download=True, transform=transforms.ToTensor())
        self.bsds = BSDS500()
        self.rng = np.random.RandomState(42)

    def __getitem__(self, index: int):
        img, target = self.svhn[index]
        return img, target

    def __len__(self):
        return len(self.svhn)

class Office31Dataset(Dataset):
    def _init_(self, root_dir, domain, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            domain (string): 'Amazon', 'webcam', or 'dslr'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, domain)
        self.domain = domain
        self.classes = os.listdir(self.root_dir)
        self.transform = transform

    def _len_(self):
        return sum(len(os.listdir(os.path.join(self.root_dir, cls))) for cls in self.classes)

    def _getitem_(self, idx):
        class_idx = idx // len(self.classes)
        class_name = self.classes[class_idx]
        img_idx = idx % len(self.classes)

        img_name = os.listdir(os.path.join(self.root_dir, class_name))[img_idx]
        img_path = os.path.join(self.root_dir, class_name, img_name)
        image = Image.open(img_path).convert('RGB')  # Ensure images are in RGB format

        if self.transform:
            image = self.transform(image)

        return image, class_idx  # Assuming class index corresponds to class_name
