from torchvision import datasets, transforms
import numpy as np
import os
from skimage.filters import gaussian
from PIL import Image


def gaussian_blur(x, severity=1):
    c = [.4, .6, 0.7, .8, 1][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c)
    return np.clip(x, 0, 1) * 255


# Assuming the gaussian_noise function is already defined as you provided
def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, .08, .09, .10, .7][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


corruption_functions = {
    'gaussian_noise': gaussian_noise,
    'gaussian_blur': gaussian_blur,
}

# Custom dataset class applying gaussian noise
class CorruptedDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None,
                 corruption_type='gaussian_noise', severity=1):
        super(CorruptedDataset, self).__init__(root, transform=None)
        self.severity = severity
        self.corruption = corruption_functions[corruption_type]
        self.transform_after_corruption = transform

    def __getitem__(self, index):
        # Get the image and label from the original ImageFolder method
        img, label = super(CorruptedDataset, self).__getitem__(index)

        # Apply the gaussian_noise function to the image
        noisy_img = self.corruption(img, self.severity)

        # Convert the numpy array back to PIL Image
        noisy_img_pil = Image.fromarray(noisy_img.astype('uint8'), 'RGB')

        # If there is a transform, apply it
        if self.transform_after_corruption is not None:
            noisy_img_pil = self.transform_after_corruption(noisy_img_pil)

        return noisy_img_pil, label

