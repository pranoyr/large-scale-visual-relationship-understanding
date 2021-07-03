import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch
from skimage.util import random_noise
import numpy as np
from PIL import Image


class GaussianNoise(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            img = Image.fromarray(random_noise(np.array(img), mode='gaussian', mean=0, var=0.001, clip=True))
            return Image.fromarray(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

