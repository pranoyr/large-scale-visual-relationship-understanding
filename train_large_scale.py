

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

def resume_model(model, optimizer):
	""" Resume model 
	"""
	checkpoint = torch.load(cfg.WEIGHT_PATH)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	print("Loaded Model ...")


dataset_train = VRDDataset(cfg.DATASET_DIR, 'train')
dataloader = DataLoader(
	dataset_train, num_workers=0, collate_fn=collater, batch_size=1)

faster_rcnn = FasterRCNN().to(cfg.DEVICE)
optimizer = optim.Adam(faster_rcnn.parameters(), lr=1e-5)

# resume model
if cfg.WEIGHT_PATH:
	resume_model(faster_rcnn, optimizer)

faster_rcnn.train()
for epoch in range(1, cfg.N_EPOCHS+1):
	loss = []
	for i, data in enumerate(dataloader):
		images, targets = data
		result, losses = faster_rcnn(images, targets)
		final_loss = losses["loss_objectness"] + losses["loss_rpn_box_reg"] + \
			losses["loss_classifier"] + losses["loss_box_reg"] + \
			losses["loss_sbj"] + losses["loss_obj"] + losses["loss_rlp"]
			

		optimizer.zero_grad()
		final_loss.backward()
		optimizer.step()

		if (i + 1) % cfg.LOG_INTERVAL == 0:	
			print(f""" Epoch {epoch} , inter
					RCNN_Loss    : {final_loss.item()}
					rpn_cls_loss   : {losses['loss_objectness'].item()}
					rpn_reg_loss   : {losses['loss_rpn_box_reg'].item()}
					box_loss 	   : {losses['loss_box_reg']}
					cls_loss       : {losses['loss_classifier']}
					sbj_loss	   : {losses['loss_sbj']}
					obj_loss	   : {losses['loss_obj']}
					sbj_acc        : {losses['acc_sbj']}
					obj_acc	       : {losses['acc_obj']}
					rlp_loss   	   : {losses['loss_rlp']}				 
					rlp_acc 	   : {losses['acc_rlp']}\n"""
					)

	state = {'state_dict': faster_rcnn.state_dict()}
	torch.save(state, os.path.join('./snapshots', f'large_scale_vrd.pth'))
	print("MODEL SAVED")
