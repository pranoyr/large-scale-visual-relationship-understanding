

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
import utils.net as net_utils 
import torchvision
import torchvision.models as models
import torchvision.models.detection._utils as det_utils
import torchvision.utils as vutils
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
from torch.jit.annotations import Dict, List, Optional, Tuple
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

from config import cfg
from datasets.vrd import VRDDataset, collater
from modelling.model import FasterRCNN
from opts import parse_opts
from utils.util import AverageMeter, Metrics, calculate_accuracy


# def train_epoch(model, dataloader, optimizer, epoch):
# 	losses_sbj = AverageMeter()
# 	losses_obj = AverageMeter()
# 	losses_rel = AverageMeter()
# 	losses_total = AverageMeter()

# 	model.train()
# 	for i, data in enumerate(dataloader):
# 		images, targets = data
# 		_, metrics = model(images, targets)
# 		final_loss = metrics["loss_objectness"] + metrics["loss_rpn_box_reg"] + \
# 			metrics["loss_classifier"] + metrics["loss_box_reg"] + \
# 			metrics["loss_sbj"] + metrics["loss_obj"] + metrics["loss_rlp"]

# 		optimizer.zero_grad()
# 		final_loss.backward()
# 		optimizer.step()

# 		losses_sbj.update(metrics["loss_sbj"].item())
# 		losses_obj.update(metrics["loss_obj"].item())
# 		losses_rel.update(metrics["loss_rlp"].item())
# 		losses_total.update(final_loss.item())

# 		if (i + 1) % 10 == 0:
# 			print(f"""RCNN_Loss    : {final_loss.item()}
# 					rpn_cls_loss   : {metrics['loss_objectness'].item()}
# 					rpn_reg_loss   : {metrics['loss_rpn_box_reg'].item()}
# 					box_loss 	   : {metrics['loss_box_reg']}
# 					cls_loss       : {metrics['loss_classifier']}
# 					sbj_loss	   : {metrics['loss_sbj']}
# 					obj_loss	   : {metrics['loss_obj']}
# 					sbj_acc        : {metrics['acc_sbj']}
# 					obj_acc	       : {metrics['acc_obj']}
# 					rlp_loss   	   : {metrics['loss_rlp']}
# 					rlp_acc 	   : {metrics['acc_rlp']}\n"""
# 				  )

# 	return losses_total.avg, losses_sbj.avg, losses_obj.avg, losses_rel.avg


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

def resume_model(args, model, optimizer):
	""" Resume model 
	"""
	checkpoint = torch.load(args.weight_path)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	print("Loaded Model ...")


def save_ckpt(step, faster_rcnn, optimizer):
	state = {'epoch': step, 'state_dict': faster_rcnn.state_dict(
	), 'optimizer_state_dict': optimizer.state_dict()}
	torch.save(state, os.path.join(
		'snapshots', f'large_scale_vrd-Epoch-{step}.pth'))
	print(f"Epoch {step} model saved!\n")


def main_worker():
	seed = 1
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	args = parse_opts()
	dataset_train = VRDDataset(cfg.DATASET_DIR, 'train')
	# dataset_val = VRDDataset(cfg.DATASET_DIR, 'test')
	dataloader = DataLoader(
		dataset_train, num_workers=cfg.SOLVER.WORKERS, collate_fn=collater, batch_size=cfg.SOLVER.BATCH_SIZE)
	# val_loader = DataLoader(
	# 	dataset_val, num_workers=cfg.WORKERS, collate_fn=collater, batch_size=cfg.BATCH_SIZE)

	faster_rcnn = FasterRCNN().to(cfg.DEVICE)

	### Optimizer ###
	# record backbone params, i.e., conv_body and box_head params
	gn_params = []
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
			if 'gn' in key:
				gn_params.append(value)
			elif 'fpn' in key or 'box_head' in key or 'box_predictor' in key or 'rpn' in key:
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
	# Learning rate of 0 is a dummy value to be set properly at the start of training
	params = [
		{'params': backbone_nonbias_params,
		 'lr': 0,
		 'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
		{'params': backbone_bias_params,
		 'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
		 'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
		{'params': prd_branch_nonbias_params,
		 'lr': 0,
		 'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
		{'params': prd_branch_bias_params,
		 'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
		 'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
		{'params': gn_params,
		 'lr': 0,
		 'weight_decay': cfg.SOLVER.WEIGHT_DECAY_GN}
	]

	if cfg.SOLVER.TYPE == "SGD":
		optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
	elif cfg.SOLVER.TYPE == "Adam":
		optimizer = torch.optim.Adam(params)

	metrics = Metrics(log_dir='tf_logs')

	# resume model
	if args.weight_path:
		resume_model(args, faster_rcnn, optimizer)

	# lr of non-backbone parameters, for commmand line outputs.
	lr = optimizer.param_groups[2]['lr']
	# lr of backbone parameters, for commmand line outputs.
	backbone_lr = optimizer.param_groups[0]['lr']

	# CHECKPOINT_PERIOD = int(cfg.TRAIN.SNAPSHOT_ITERS / cfg.NUM_GPUS)
	CHECKPOINT_PERIOD = cfg.SOLVER.MAX_ITER / 20000

	# Set index for decay steps
	decay_steps_ind = None
	for i in range(1, len(cfg.SOLVER.STEPS)):
		if cfg.SOLVER.STEPS[i] >= args.start_step:
			decay_steps_ind = i
			break
	if decay_steps_ind is None:
		decay_steps_ind = len(cfg.SOLVER.STEPS)

	# training_stats = TrainingStats(
	# 	args,
	# 	args.disp_interval,
	# 	tblogger if args.use_tfboard and not args.no_save else None)

	print('Training starts !')
	step = args.start_step
	for step in range(args.start_step, cfg.SOLVER.MAX_ITER):

		# Warm up
		if step < cfg.SOLVER.WARM_UP_ITERS:
			method = cfg.SOLVER.WARM_UP_METHOD
			if method == 'constant':
				warmup_factor = cfg.SOLVER.WARM_UP_FACTOR
			elif method == 'linear':
				alpha = step / cfg.SOLVER.WARM_UP_ITERS
				warmup_factor = cfg.SOLVER.WARM_UP_FACTOR * \
					(1 - alpha) + alpha
			else:
				raise KeyError(
					'Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
			lr_new = cfg.SOLVER.BASE_LR * warmup_factor
			net_utils.update_learning_rate_rel(optimizer, lr, lr_new)
			lr = optimizer.param_groups[2]['lr']
			backbone_lr = optimizer.param_groups[0]['lr']
			assert lr == lr_new
		elif step == cfg.SOLVER.WARM_UP_ITERS:
			net_utils.update_learning_rate_rel(
				optimizer, lr, cfg.SOLVER.BASE_LR)
			lr = optimizer.param_groups[2]['lr']
			backbone_lr = optimizer.param_groups[0]['lr']
			assert lr == cfg.SOLVER.BASE_LR

		# Learning rate decay
		if decay_steps_ind < len(cfg.SOLVER.STEPS) and \
				step == cfg.SOLVER.STEPS[decay_steps_ind]:
			print('Decay the learning on step %d', step)
			lr_new = lr * cfg.SOLVER.GAMMA
			net_utils.update_learning_rate_rel(optimizer, lr, lr_new)
			lr = optimizer.param_groups[2]['lr']
			backbone_lr = optimizer.param_groups[0]['lr']
			assert lr == lr_new
			decay_steps_ind += 1

		# training_stats.IterTic()
		optimizer.zero_grad()
		for inner_iter in range(args.iter_size):
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

			# training_stats.UpdateIterStats(net_outputs, inner_iter)
			loss.backward()
		optimizer.step()
		# training_stats.IterToc()

		# training_stats.LogIterStats(step, lr, backbone_lr)

		if (step+1) % CHECKPOINT_PERIOD == 0:
			save_ckpt(step, faster_rcnn, optimizer)

	# ---- Training ends ----
	# Save last checkpoint
	save_ckpt(step, faster_rcnn, optimizer)

# for epoch in range(1, cfg.N_EPOCHS+1):
# 	train_metrics = train_epoch(
# 			faster_rcnn, train_loader, optimizer, epoch)

# 	if epoch % 1 == 0:
# 		metrics.log_metrics(train_metrics, epoch)

# 		state = {'epoch': epoch, 'state_dict': faster_rcnn.state_dict(
# 		), 'optimizer_state_dict': optimizer.state_dict()}
# 		torch.save(state, os.path.join(
# 			'snapshots', f'large_scale_vrd-Epoch-{epoch}.pth'))
# 		print(f"Epoch {epoch} model saved!\n")


if __name__ == "__main__":
	main_worker()
