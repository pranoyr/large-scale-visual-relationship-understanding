

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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


def train_epoch(model, dataloader, optimizer, epoch):
	losses_sbj = AverageMeter()
	losses_obj = AverageMeter()
	losses_rel = AverageMeter()
	losses_total = AverageMeter()

	model.train()
	for i, data in enumerate(dataloader):
		images, targets = data
		_, metrics = model(images, targets)
		final_loss = metrics["loss_objectness"] + metrics["loss_rpn_box_reg"] + \
			metrics["loss_classifier"] + metrics["loss_box_reg"] + \
			metrics["loss_sbj"] + metrics["loss_obj"] + metrics["loss_rlp"]

		optimizer.zero_grad()
		final_loss.backward()
		optimizer.step()

		losses_sbj.update(metrics["loss_sbj"].item())
		losses_obj.update(metrics["loss_obj"].item())
		losses_rel.update(metrics["loss_rlp"].item())
		losses_total.update(final_loss.item())
		if (i + 1) % 10 == 0:
			print(f"""RCNN_Loss    : {final_loss.item()}
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
	losses = {}
	losses['total_loss'] = losses_total.avg
	losses['sbj_loss'] = losses_sbj.avg
	losses['obj_loss'] = losses_obj.avg
	losses['rel_loss'] = losses_rel.avg
	return losses


# def val_epoch(model, dataloader):
# 	losses_sbj = AverageMeter()
# 	losses_obj = AverageMeter()
# 	losses_rel = AverageMeter()
# 	losses_total = AverageMeter()

# 	model.eval()
# 	for _, data in enumerate(dataloader):
# 		images, targets = data
# 		with torch.no_grad():
# 			_, metrics = model(images, targets)
# 		final_loss = metrics["loss_objectness"] + metrics["loss_rpn_box_reg"] + \
# 			metrics["loss_classifier"] + metrics["loss_box_reg"] + \
# 			metrics["loss_sbj"] + metrics["loss_obj"] + metrics["loss_rlp"]

# 		losses_sbj.update(metrics["loss_sbj"].item())
# 		losses_obj.update(metrics["loss_obj"].item())
# 		losses_rel.update(metrics["loss_rlp"].item())
# 		losses_total.update(final_loss.item())

# 	return losses_total.avg, losses_sbj.avg, losses_obj.avg, losses_rel.avg

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
		'snapshots', f'large_scale_vrd-Epoch-{epoch}.pth'))
	print(f"Epoch {epoch} model saved!\n")

		
def main_worker():
	seed = 1
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	opt = parse_opts()
	dataset_train = VRDDataset(cfg.DATASET_DIR, 'train')
	# dataset_val = VRDDataset(cfg.DATASET_DIR, 'test')
	train_loader = DataLoader(
		dataset_train, num_workers=opt.num_workers, collate_fn=collater, batch_size=opt.batch_size)
	# val_loader = DataLoader(
	# 	dataset_val, num_workers=cfg.WORKERS, collate_fn=collater, batch_size=cfg.BATCH_SIZE)

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
		 'lr': cfg.TRAIN.LEARNING_RATE,
		 'weight_decay': cfg.TRAIN.WEIGHT_DECAY},
		{'params': backbone_bias_params,
		 'lr': cfg.TRAIN.LEARNING_RATE * (cfg.TRAIN.DOUBLE_BIAS + 1),
		 'weight_decay': cfg.TRAIN.WEIGHT_DECAY if cfg.TRAIN.BIAS_DECAY else 0},
		{'params': prd_branch_nonbias_params,
		 'lr': cfg.TRAIN.LEARNING_RATE,
		 'weight_decay': cfg.TRAIN.WEIGHT_DECAY},
		{'params': prd_branch_bias_params,
		 'lr': cfg.TRAIN.LEARNING_RATE * (cfg.TRAIN.DOUBLE_BIAS + 1),
		 'weight_decay': cfg.TRAIN.WEIGHT_DECAY if cfg.TRAIN.BIAS_DECAY else 0},
	]

	optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

	lr = optimizer.param_groups[2]['lr']  # lr of non-backbone parameters, for commmand line outputs.

	if cfg.TRAIN.TYPE == "ADAM":
		optimizer = torch.optim.Adam(params)
		
	elif cfg.TRAIN.TYPE == "SGD":
		optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

	metrics = Metrics(log_dir='tf_logs')

	# resume model
	if opt.weight_path:
		resume_model(opt, faster_rcnn, optimizer)

	for epoch in range(1, opt.n_epochs):
		losses = train_epoch(
				faster_rcnn, train_loader, optimizer, epoch)

		if epoch % 5 == 0:
			lr_new = lr * cfg.TRAIN.GAMMA
			net_utils.update_learning_rate_att(optimizer, lr, lr_new)
			lr = optimizer.param_groups[2]['lr']

		if epoch % 1 == 0:
			metrics.log_metrics(losses, epoch)
			save_model(faster_rcnn, optimizer, epoch)
			
if __name__ == "__main__":
	main_worker()
