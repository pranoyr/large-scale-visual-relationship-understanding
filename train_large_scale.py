

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



def val_epoch(model, dataloader):
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

		losses_sbj.update(metrics["loss_sbj"].item())
		losses_obj.update(metrics["loss_obj"].item())
		losses_rel.update(metrics["loss_rlp"].item())
		losses_total.update(final_loss.item())

	losses = {}
	losses['total_loss'] = losses_total.avg
	losses['sbj_loss'] = losses_sbj.avg
	losses['obj_loss'] = losses_obj.avg
	losses['rel_loss'] = losses_rel.avg
	return losses

def load_from_ckpt(opt, model):
	""" Loading model from checkpoint
	"""
	checkpoint = torch.load(opt.weight_path)
	model.load_state_dict(checkpoint['state_dict'])
	print("Loaded Model ...")

def load_optmizer(opt, optimizer):
	""" loading optmizer 
	"""
	checkpoint = torch.load(opt.weight_path)
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	print("Loaded Optimizer ...")


def save_model(model, optimizer, step):
	state = {'step': step, 'state_dict': model.state_dict(
			), 'optimizer_state_dict': optimizer.state_dict()}
	torch.save(state, os.path.join(
		'snapshots', f'large_scale_vrd-iter-{step}.pth'))
	
	
def main_worker():
	seed = 1
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	opt = parse_opts()
	dataset_train = VRDDataset(cfg.DATASET_DIR, 'train')
	dataset_val = VRDDataset(cfg.DATASET_DIR, 'test')
	train_loader = DataLoader(
		dataset_train, num_workers=opt.num_workers, collate_fn=collater, batch_size=opt.batch_size, shuffle=True)
	val_loader = DataLoader(
		dataset_val, num_workers=opt.num_workers, collate_fn=collater, batch_size=opt.batch_size, shuffle=False)

	print(f"Training dataset size : {len(train_loader.dataset)}")
	print(f"Validation dataset size : {len(val_loader.dataset)}")

	dataiterator = iter(train_loader)

	faster_rcnn = FasterRCNN()

	# loading model from a ckpt
	if opt.weight_path:
		load_from_ckpt(opt, faster_rcnn)
	faster_rcnn.to(cfg.DEVICE)

	
	lr = cfg.TRAIN.LEARNING_RATE
	#tr_momentum = cfg.TRAIN.MOMENTUM
	#tr_momentum = args.momentum

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

	if cfg.TRAIN.TYPE == "ADAM":
		optimizer = torch.optim.Adam(params)
		
	elif cfg.TRAIN.TYPE == "SGD":
		optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

	if opt.weight_path:
		load_optmizer(opt, optimizer)

	# scheduler 
	if opt.scheduler == "plateau":
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
	elif opt.scheduler == "multi_step":
		scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[8, 12])
	elif opt.scheduler == "step_lr":
		scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1)

	metrics = Metrics(log_dir='tf_logs')

	losses_sbj = AverageMeter()
	losses_obj = AverageMeter()
	losses_rel = AverageMeter()
	losses_total = AverageMeter()

	faster_rcnn.train()
	max_iter = 125446
	for step in range(1, max_iter):
		try:
			input_data = next(dataiterator)
		except StopIteration:
			dataiterator = iter(train_loader)
			input_data = next(dataiterator)

		images, targets = input_data
		_, metrics = faster_rcnn(images, targets)
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

		if (step) % 10 == 0:
			print(f"""Iteration    : {step}
					RCNN_Loss	   : {final_loss.item()}
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


		if step % 500 == 0:
			train_losses = {}
			train_losses['total_loss'] = losses_total.avg
			train_losses['sbj_loss'] = losses_sbj.avg
			train_losses['obj_loss'] = losses_obj.avg
			train_losses['rel_loss'] = losses_rel.avg
			val_losses = val_epoch(faster_rcnn, val_loader)

			scheduler.step(val_losses['total_loss'])
			lr = optimizer.param_groups[0]['lr']  

			# write summary
			metrics.log_metrics(train_losses, val_losses, step, lr)
			save_model(faster_rcnn, optimizer, step)
			print(f"Saved model")

	
			# avg_loss.reset()
			# avg_clf_loss.reset()
			# avg_reg_loss.reset()
			

	# for step in range(1, max_iter):
	# 	try:
	# 		input_data = next(dataiterator)
	# 	except StopIteration:
	# 		dataiterator = iter(train_loader)
	# 		input_data = next(dataiterator)

	# 	images, targets = input_data
		

	# 	confidence, locations = net(images)

	# 	regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
	# 	loss = regression_loss + classification_loss

	# 	avg_loss.update(loss.item(), images.size(0))
	# 	avg_clf_loss.update(classification_loss.item(), images.size(0))
	# 	avg_reg_loss.update(regression_loss.item(), images.size(0))

	# 	optimizer.zero_grad()
	# 	loss.backward()
	# 	optimizer.step()
	# 	if args.scheduler != 'plateau':
	# 			scheduler.step()


	# 	if step % 10 == 0:
	# 		logging.info(
	# 			f"Iter: {step}, " +
	# 			f"Loss: {avg_loss.avg:.4f}, " +
	# 			f"Regression Loss {avg_reg_loss.avg:.4f}, " +
	# 			f"Classification Loss: {avg_clf_loss.avg:.4f}"
	# 		)


	# 	if (step) % 1000 == 0:
	# 		val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, device)
	# 		if args.scheduler == 'plateau':
	# 			scheduler.step(val_loss)

	# 		logging.info(
	# 			f"Iter: {step}, " +
	# 			f"Average Loss: {avg_loss.avg:.4f}, " +
	# 			f"Average Regression Loss {avg_reg_loss.avg:.4f}, " +
	# 			f"Average Classification Loss: {avg_clf_loss.avg:.4f}"
	# 		)

	# 		logging.info(
	# 			f"Iter: {step}, " +
	# 			f"Validation Loss: {val_loss:.4f}, " +
	# 			f"Validation Regression Loss {val_regression_loss:.4f}, " +
	# 			f"Validation Classification Loss: {val_classification_loss:.4f}"
	# 		)

	# 		lr = optimizer.param_groups[0]['lr']  

	# 		# write summary
	# 		summary_writer.add_scalar(
	# 			'losses/val_loss', val_loss, global_step=step)
	# 		summary_writer.add_scalar(
	# 			'losses/val_regression_loss', val_regression_loss, global_step=step)
	# 		summary_writer.add_scalar(
	# 			'losses/val_classification_loss', val_classification_loss, global_step=step)
	# 		summary_writer.add_scalar(
	# 			'losses/train_loss', avg_loss.avg, global_step=step)
	# 		summary_writer.add_scalar(
	# 			'losses/train_regression_loss', avg_reg_loss.avg, global_step=step)
	# 		summary_writer.add_scalar(
	# 			'losses/train_classification_loss', avg_clf_loss.avg, global_step=step)
	# 		summary_writer.add_scalar(
	# 			'lr', lr, global_step=step)

	# 		avg_loss.reset()
	# 		avg_clf_loss.reset()
	# 		avg_reg_loss.reset()
	# 		net.train()

	# 		model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Iter-{step}-Loss-{val_loss}.pth")

	# 		net.save(model_path)
	# 		logging.info(f"Saved model {model_path}")



	# for epoch in range(1, opt.n_epochs):
	# 	train_losses = train_epoch(
	# 			faster_rcnn, train_loader, optimizer, epoch)
				
	# 	val_losses = val_epoch(faster_rcnn, val_loader)

	# 	if opt.scheduler == "plateau":
	# 		scheduler.step(val_losses['total_loss'])
	# 	else:
	# 		scheduler.step()
				
	# 	lr = optimizer.param_groups[2]['lr']  

	# 	# if epoch % 5 == 0:
	# 		# lr_new = lr * cfg.TRAIN.GAMMA
	# 		# net_utils.update_learning_rate_rel(optimizer, lr, lr_new)

	# 	if epoch % 1 == 0:
	# 		metrics.log_metrics(train_losses, val_losses, epoch, lr)
	# 		save_model(faster_rcnn, optimizer, epoch)
		
if __name__ == "__main__":
	main_worker()
