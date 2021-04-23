

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
import utils.net as net_utils
import torchvision.models.detection._utils as det_utils
import torchvision.utils as vutils
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
from torch.jit.annotations import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import (FastRCNNPredictor,
													  GeneralizedRCNNTransform,
													  MultiScaleRoIAlign,
													  TwoMLPHead)
from torchvision.models.detection.rpn import (AnchorGenerator,
											  RegionProposalNetwork, RPNHead)
from torchvision.models.resnet import resnet101
from torchvision.ops import boxes as box_ops

from config import cfg
from datasets.vrd import VRDDataset, collater
from modelling.model import FasterRCNN
from opts import parse_opts
from utils.util import AverageMeter, Metrics, calculate_accuracy


def resume_model(opt, model, optimizer):
	""" Resume model 
	"""
	checkpoint = torch.load(opt.weight_path)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	print("Loaded Model ...")


def save_model(model, optimizer, epoch):
	state = {'epoch': epoch, 'state_dict': model.state_dict(
			), 'optimizer_state_dict': optimizer.state_dict()}
	torch.save(state, os.path.join(
		'snapshots', f'large_scale_vrd.pth'))
	print(f"Epoch {epoch} model saved!\n")

		
def main_worker():
	seed = 1
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	opt = parse_opts()
	dataset_train = VRDDataset(cfg.DATASET_DIR, 'train')
	dataloader = DataLoader(
		dataset_train, num_workers=opt.num_workers, collate_fn=collater, batch_size=opt.batch_size)
	dataiterator = iter(dataloader)

	faster_rcnn = FasterRCNN().to(cfg.DEVICE)
	
	### Optimizer ###
	# record backbone params, i.e., conv_body and box_head params
	backbone_bias_params = []
	backbone_bias_param_names = []
	prd_branch_bias_params = []
	prd_branch_bias_param_names = []
	backbone_nonbias_params = []
	backbone_nonbias_param_names = []
	prd_branch_nonbias_params = []
	prd_branch_nonbias_param_names = []
	for key, value in dict(faster_rcnn.named_parameters()).items():
		if value.requires_grad:
			if 'fpn' in key or 'box_head' in key or 'box_predictor' in key or 'rpn' in key:
				if 'bias' in key:
					backbone_bias_params.append(value)
					backbone_bias_param_names.append(key)
				else:
					backbone_nonbias_params.append(value)
					backbone_nonbias_param_names.append(key)
			else:
				if 'bias' in key:
					prd_branch_bias_params.append(value)
					prd_branch_bias_param_names.append(key)
				else:
					prd_branch_nonbias_params.append(value)
					prd_branch_nonbias_param_names.append(key)
	params = [
		 {'params': backbone_nonbias_params,
		 'lr': 0,
		 'weight_decay': cfg.TRAIN.WEIGHT_DECAY},
		{'params': backbone_bias_params,
		 'lr': 0 * (cfg.TRAIN.DOUBLE_BIAS + 1),
		 'weight_decay': cfg.TRAIN.WEIGHT_DECAY if cfg.TRAIN.BIAS_DECAY else 0},
		{'params': prd_branch_nonbias_params,
		 'lr': 0,
		 'weight_decay': cfg.TRAIN.WEIGHT_DECAY},
		{'params': prd_branch_bias_params,
		 'lr': 0 * (cfg.TRAIN.DOUBLE_BIAS + 1),
		 'weight_decay': cfg.TRAIN.WEIGHT_DECAY if cfg.TRAIN.BIAS_DECAY else 0}
	]

	if cfg.TRAIN.TYPE == "SGD":
		optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
	elif cfg.TRAIN.TYPE == "Adam":
		optimizer = torch.optim.Adam(params)

	lr = optimizer.param_groups[2]['lr']  # lr of non-backbone parameters, for commmand line outputs.
	backbone_lr = optimizer.param_groups[0]['lr']  # lr of backbone parameters, for commmand line outputs.

	# Set index for decay steps
	decay_steps_ind = None
	for i in range(1, len(cfg.TRAIN.STEPS)):
		if cfg.TRAIN.STEPS[i] >= 0:
			decay_steps_ind = i
			break
	if decay_steps_ind is None:
		decay_steps_ind = len(cfg.TRAIN.STEPS)

	print('Training starts !')
	for step in range(0, cfg.TRAIN.MAX_ITER):
		# Warm up
		if step < cfg.TRAIN.WARM_UP_ITERS:
			method = cfg.TRAIN.WARM_UP_METHOD
			if method == 'constant':
				warmup_factor = cfg.TRAIN.WARM_UP_FACTOR
			elif method == 'linear':
				alpha = step / cfg.TRAIN.WARM_UP_ITERS
				warmup_factor = cfg.TRAIN.WARM_UP_FACTOR * (1 - alpha) + alpha
			else:
				raise KeyError('Unknown TRAIN.WARM_UP_METHOD: {}'.format(method))
			lr_new = cfg.TRAIN.LEARNING_RATE * warmup_factor
			net_utils.update_learning_rate_rel(optimizer, lr, lr_new)
			lr = optimizer.param_groups[2]['lr']
			backbone_lr = optimizer.param_groups[0]['lr']
			assert lr == lr_new
		elif step == cfg.TRAIN.WARM_UP_ITERS:
			net_utils.update_learning_rate_rel(optimizer, lr, cfg.TRAIN.LEARNING_RATE)
			lr = optimizer.param_groups[2]['lr']
			backbone_lr = optimizer.param_groups[0]['lr']
			assert lr == cfg.TRAIN.LEARNING_RATE

		# Learning rate decay
		if decay_steps_ind < len(cfg.TRAIN.STEPS) and \
				step == cfg.TRAIN.STEPS[decay_steps_ind]:
			print('Decay the learning on step %d', step)
			lr_new = lr * cfg.TRAIN.GAMMA
			net_utils.update_learning_rate_rel(optimizer, lr, lr_new)
			lr = optimizer.param_groups[2]['lr']
			backbone_lr = optimizer.param_groups[0]['lr']
			assert lr == lr_new
			decay_steps_ind += 1

		optimizer.zero_grad()
		for inner_iter in range(1):
			try:
				input_data = next(dataiterator)
			except StopIteration:
				dataiterator = iter(dataloader)
				input_data = next(dataiterator)
			
			images, targets = input_data
			_	, metrics = faster_rcnn(images, targets)
			loss = metrics["loss_objectness"] + metrics["loss_rpn_box_reg"] + \
				metrics["loss_classifier"] + metrics["loss_box_reg"] + \
				metrics["loss_sbj"] + \
				metrics["loss_obj"] + metrics["loss_rlp"]

			print(f""" Iteration {step}/{cfg.TRAIN.MAX_ITER}
					RCNN_Loss      : {loss.item()}
					rpn_cls_loss   : {metrics['loss_objectness'].item()}
					rpn_reg_loss   : {metrics['loss_rpn_box_reg'].item()}
					box_loss 	   : {metrics['loss_box_reg']}
					cls_loss       : {metrics['loss_classifier']}
					sbj_loss	   : {metrics['loss_sbj']}
					obj_loss	   : {metrics['loss_obj']}
					sbj_acc        : {metrics['acc_sbj']}
					obj_acc	       : {metrics['acc_obj']}
					rlp_loss   	   : {metrics['loss_rlp']}
					rlp_acc 	   : {metrics['acc_rlp']}\n"""
				)
			loss.backward()
		optimizer.step()


		if (step+1) % 500 == 0:
			save_model(faster_rcnn, optimizer, step)

	# ---- Training ends ----
	# Save last checkpoint
	save_model(faster_rcnn, optimizer, step)


if __name__ == '__main__':
	main_worker()

	
