

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
from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import (AnchorGenerator,
                                              RegionProposalNetwork, RPNHead)
from torchvision.models.resnet import resnet101
from torchvision.ops import boxes as box_ops


class RPN(nn.Module):
	def __init__(self):
		super(RPN, self).__init__()
		# Define FPN
		anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
		aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
		# Generate anchor boxes
		anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
		# Define RPN Head
		rpn_head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
		RPN_PRE_NMS_TOP_N = dict(training=cfg.RPN_PRE_NMS_TOP_N_TRAIN,
								 testing=cfg.RPN_PRE_NMS_TOP_N_TEST)
		RPN_POST_NMS_TOP_N = dict(
			training=cfg.RPN_POST_NMS_TOP_N_TRAIN, testing=cfg.RPN_POST_NMS_TOP_N_TEST)

		# Create RPN
		self.rpn = RegionProposalNetwork(
			anchor_generator, rpn_head,
			cfg.RPN_FG_IOU_THRESH, cfg.RPN_BG_IOU_THRESH,
			cfg.RPN_BATCH_SIZE_PER_IMAGE, cfg.RPN_POSITIVE_FRACTION,
			RPN_PRE_NMS_TOP_N, RPN_POST_NMS_TOP_N, cfg.RPN_NMS_THRESH)
	
	def forward(self, images, fpn_feature_maps, targets=None):
		if targets is not None:
			boxes, losses = self.rpn(images, fpn_feature_maps, targets)
		else:
			boxes, losses = self.rpn(images, fpn_feature_maps)
		return boxes, losses, fpn_feature_maps
