# This file contains 2 misc. functions we needed, mainly for adaptations involving MNIST.

from PIL import Image

import numpy as np
from PIL import Image

class GrayscaleToRgb:
    '''
        Converts 1 channel image to 3 channel image.
    '''
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)

class PadSize:
    '''
        Pads the image to the target size.
    '''

    def __init__(self, target_size=(32, 32)):
        self.target_size = target_size

    def __call__(self, image):
        image_np = np.array(image)
        pad_width = ((self.target_size[0] - image_np.shape[0]) // 2, (self.target_size[1] - image_np.shape[1]) // 2)
        
        padded_image = np.pad(image_np, ((pad_width[0], pad_width[0]), (pad_width[1], pad_width[1])), 'constant')
        padded_image_pil = Image.fromarray(padded_image)
        return padded_image_pil


