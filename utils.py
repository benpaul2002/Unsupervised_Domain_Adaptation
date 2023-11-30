from PIL import Image

import numpy as np
from PIL import Image

class GrayscaleToRgb:
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)

class PadSize:
    def __init__(self, target_size=(32, 32)):
        self.target_size = target_size

    def __call__(self, image):
        image_np = np.array(image)
        pad_width = ((self.target_size[0] - image_np.shape[0]) // 2, (self.target_size[1] - image_np.shape[1]) // 2)
        
        padded_image = np.pad(image_np, ((pad_width[0], pad_width[0]), (pad_width[1], pad_width[1])), 'constant')
        padded_image_pil = Image.fromarray(padded_image)
        return padded_image_pil


