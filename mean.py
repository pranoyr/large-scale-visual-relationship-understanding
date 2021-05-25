import math
import os
import pdb
import random
import time
from collections import OrderedDict
import utils.net as net_utils

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.models.detection._utils as det_utils
import torchvision.utils as vutils
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
from torch.jit.annotations import Dict, List, Optional, Tuple
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import (FastRCNNPredictor,
                                                      GeneralizedRCNNTransform,
                                                      MultiScaleRoIAlign,
                                                      TwoMLPHead)
from torchvision.models.detection.rpn import (AnchorGenerator,
                                              RegionProposalNetwork, RPNHead)
from torchvision.models.resnet import resnet101
from torchvision.ops import boxes as box_ops
from dataset import get_training_data, get_validation_data

from config import cfg
from datasets.vrd import collater
from modelling.model import FasterRCNN
from opts import parse_opts
from utils.util import AverageMeter, Metrics, ProgressMeter
from torchvision import transforms, utils


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
    images = _resize_image_and_masks(images)
    images = images.view(images.size(0), images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    nb_samples += images.size(0)

mean /= nb_samples
std /= nb_samples

print(mean)
print(std)

    
