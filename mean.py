import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import get_training_data, get_validation_data

from config import cfg
from datasets.vrd import collater
from opts import parse_opts


mean = 0.
std = 0.
nb_samples = 0.

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


opt = parse_opts()
train_data = get_training_data(cfg)
val_data = get_validation_data(cfg)
train_loader = DataLoader(
    train_data, num_workers=opt.num_workers, collate_fn=collater, batch_size=1, shuffle=True)

def _resize_image_and_masks(image, self_min_size=800, self_max_size=1333):
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))
    max_size = float(torch.max(im_shape))
    scale_factor = self_min_size / min_size
    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode='bilinear', recompute_scale_factor=True,
        align_corners=False)[0]
    return image
       

for data in train_loader:
    images, targets = data
    images = images[0]
    images = _resize_image_and_masks(images).unsqueeze(0)
    images = images.view(images.size(0), images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    nb_samples += images.size(0)

mean /= nb_samples
std /= nb_samples

print(mean)
print(std)

    
