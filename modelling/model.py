import math
import os
import pdb
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.models.detection._utils as det_utils
import torchvision.utils as vutils
from config import cfg
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
from torch.jit.annotations import Dict, List, Optional, Tuple
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import (FastRCNNPredictor,
                                                      GeneralizedRCNNTransform,
                                                      MultiScaleRoIAlign,
                                                      TwoMLPHead)
from torchvision.models.resnet import resnet101
from torchvision.ops import boxes as box_ops
from utils.boxes import postprocess

from .roi_head import RoIHeads
from .rpn import RPN


class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        # Define FPN
        self.fpn = resnet_fpn_backbone(
            backbone_name='resnet101', pretrained=True, trainable_layers=5)
        self.rpn = RPN()
        self.roi_heads = RoIHeads()

        # transform parameters
        min_size = 800
        max_size = 1333
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std)

    def flatten_targets(self, targets):
        gth_list = []
        for target in targets:
            gt = {}
            gt["boxes"] = target["boxes"].view(-1, 4)
            gt["labels"] = target["labels"].view(-1)
            gt["preds"] = target["preds"].view(-1)
            gth_list.append(gt)
        return gth_list

    def unflatten_targets(self, targets):
        gth_list = []
        for target in targets:
            gt = {}
            gt["boxes"] = target["boxes"].view(-1, 2, 4)
            gt["labels"] = target["labels"].view(-1, 2)
            gt["preds"] = target["preds"].view(-1)
            gth_list.append(gt)
        return gth_list

    def forward(self, images, targets=None):
    
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        if self.training:
            targets = self.flatten_targets(targets)
        images, targets = self.transform(images, targets)

        fpn_feature_maps = self.fpn(images.tensors.to(cfg.DEVICE))

        if self.training:
            proposals, rpn_losses, fpn_feature_maps = self.rpn(
                images, fpn_feature_maps, targets)
            targets = self.unflatten_targets(targets)
            detections, detector_losses = self.roi_heads(
                fpn_feature_maps, proposals, images.image_sizes, targets)
            losses = {}
            losses.update(detector_losses)
            losses.update(rpn_losses)
        else:
            losses = {}
            proposals, rpn_losses, fpn_feature_maps = self.rpn(
                images, fpn_feature_maps)
            detections, detector_losses = self.roi_heads(
                fpn_feature_maps, proposals, images.image_sizes)
            detections = postprocess(
                detections, images.image_sizes, original_image_sizes)
        return detections, losses
