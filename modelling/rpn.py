

from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead, AnchorGenerator
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models.detection._utils as  det_utils
from torchvision.ops import boxes as box_ops
import os

from collections import OrderedDict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
import time
import pdb
from torchvision.models.resnet import resnet101
import torchvision
import math
from torch.jit.annotations import Optional, List, Dict, Tuple
from config import cfg


class RPN(nn.Module):
	def __init__(self):
		super(RPN, self).__init__()
		# Define FPN
		anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
		aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
		# Generate anchor boxes
		anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
		# Define RPN Head
		# rpn_head = RPNHead(256, 9)
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

	def prepare_gt_for_rpn(self, targets):
		gth_list = []
		for target in targets:
			gt = {}
			gt["boxes"] = target["boxes"].view(-1,4)
			gt["labels"] = target["labels"].view(-1)
			gth_list.append(gt)
		return gth_list
	
	def forward(self, images, fpn_feature_maps, targets=None):
		# l = torch.FloatTensor([[1,2,3,4],[1,2,3,4]])
		# targets = [{"boxes":l},{"boxes":l}]
		# targets = [{i: index for i, index in enumerate(l)}]
		targets = self.prepare_gt_for_rpn(targets)
		
		if self.training:
			boxes, losses = self.rpn(images, fpn_feature_maps, targets)
		else:
			boxes, losses = self.rpn(images, fpn_feature_maps)
		return boxes, losses, fpn_feature_maps
