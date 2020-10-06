

from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead, AnchorGenerator
# from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 
# from datasets.pascal_voc import VOCDataset, collater
from datasets.vrd import VRDDataset, collater
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models.detection._utils as  det_utils
from torchvision.ops import boxes as box_ops
import os
from modelling.model import FasterRCNN

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
from torchvision.models.detection.faster_rcnn import MultiScaleRoIAlign, TwoMLPHead, FastRCNNPredictor 
from config import cfg


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_train = VRDDataset(cfg.DATASET_DIR, 'train')
dataloader = DataLoader(
	dataset_train, num_workers=0, collate_fn=collater, batch_size=1)


faster_rcnn = FasterRCNN().to(DEVICE)
optimizer = optim.Adam(faster_rcnn.parameters(), lr=1e-5)
faster_rcnn.train()

for epoch in range(1, cfg.N_EPOCHS+1):
	loss = []
	for i, data in enumerate(dataloader):
		images, targets = data
		result, losses = faster_rcnn(images, targets)
		final_loss = losses["loss_objectness"] + losses["loss_rpn_box_reg"] + \
			losses["loss_classifier"] + losses["loss_box_reg"] + \
			losses["loss_sbj"] + losses["loss_obj"] + losses["loss_rlp"]
			

		loss.append(final_loss.item())

		optimizer.zero_grad()
		final_loss.backward()
		optimizer.step()
		print(f"""RCNN_Loss    : {final_loss.item()}\n\
				rpn_cls_loss   : {losses['loss_objectness'].item()}\n\
				rpn_reg_loss   : {losses['loss_rpn_box_reg'].item()}\n\
				box_loss 	   : {losses['loss_box_reg']}\n\
				cls_loss       : {losses['loss_classifier']}\n\
				sbj_loss	   : {losses['loss_sbj']}\n\
				obj_loss	   : {losses['loss_obj']}\n\
				sbj_acc        : {losses['acc_sbj']}\n\
				obj_acc	       : {losses['acc_obj']}\n\
			    rlp_loss   	   : {losses['loss_rlp']}\n\					 
				rlp_acc 	   : {losses['acc_rlp']}"""
				)

	loss = torch.tensor(loss, dtype=torch.float32)
	print(f'loss : {torch.mean(loss)}')
	# scheduler.step(torch.mean(loss))

	state = {'state_dict': faster_rcnn.state_dict()}
	torch.save(state, os.path.join('./snapshots', f'faster_rcnn.pth'))
	print("model saved")

