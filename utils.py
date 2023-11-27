from PIL import Image

import numpy as np
from PIL import Image

class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)

class PadSize:
    """Pad images to a target size"""
    def __init__(self, target_size=(32, 32)):
        self.target_size = target_size

    def __call__(self, image):
        # Convert PIL Image to numpy array
        image_np = np.array(image)

        # Calculate padding
        pad_width = ((self.target_size[0] - image_np.shape[0]) // 2, 
                     (self.target_size[1] - image_np.shape[1]) // 2)
        
        # Pad image
        padded_image = np.pad(image_np, ((pad_width[0], pad_width[0]), 
                                         (pad_width[1], pad_width[1])), 
                              'constant')

        # Convert numpy array back to PIL Image
        padded_image_pil = Image.fromarray(padded_image)

        return padded_image_pil


