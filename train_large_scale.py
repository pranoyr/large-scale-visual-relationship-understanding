

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

	return losses_total.avg, losses_sbj.avg, losses_obj.avg, losses_rel.avg


def val_epoch(model, dataloader):
	losses_sbj = AverageMeter()
	losses_obj = AverageMeter()
	losses_rel = AverageMeter()
	losses_total = AverageMeter()

	model.eval()
	for _, data in enumerate(dataloader):
		images, targets = data
		_, metrics = model(images, targets)
		final_loss = metrics["loss_objectness"] + metrics["loss_rpn_box_reg"] + \
			metrics["loss_classifier"] + metrics["loss_box_reg"] + \
			metrics["loss_sbj"] + metrics["loss_obj"] + metrics["loss_rlp"]

		losses_sbj.update(metrics["loss_sbj"].item())
		losses_obj.update(metrics["loss_obj"].item())
		losses_rel.update(metrics["loss_rlp"].item())
		losses_total.update(final_loss.item())

	return losses_total.avg, losses_sbj.avg, losses_obj.avg, losses_rel.avg

def resume_model(opt, model, optimizer):
	""" Resume model 
	"""
	checkpoint = torch.load(opt.weight_path)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	print("Loaded Model ...")


def main_worker():
	seed = 1
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	opt = parse_opts()
	dataset_train = VRDDataset(cfg.DATASET_DIR, 'train')
	dataset_val = VRDDataset(cfg.DATASET_DIR, 'test')
	train_loader = DataLoader(
		dataset_train, num_workers=cfg.WORKERS, collate_fn=collater, batch_size=cfg.BATCH_SIZE)
	val_loader = DataLoader(
		dataset_val, num_workers=cfg.WORKERS, collate_fn=collater, batch_size=cfg.BATCH_SIZE)

	faster_rcnn = FasterRCNN().to(cfg.DEVICE)
	optimizer = optim.Adam(faster_rcnn.parameters(), lr=cfg.LR_RATE, weight_decay=cfg.WEIGHT_DECAY)
	metrics = Metrics(log_dir='tf_logs')

	# resume model
	if opt.weight_path:
		resume_model(opt, faster_rcnn, optimizer)

	for epoch in range(1, cfg.N_EPOCHS+1):
		train_metrics = train_epoch(
				faster_rcnn, train_loader, optimizer, epoch)

		if epoch % 1 == 0:
			val_metrics = val_epoch(faster_rcnn, val_loader)
			metrics.log_metrics(train_metrics, val_metrics, epoch)

			state = {'epoch': epoch, 'state_dict': faster_rcnn.state_dict(
			), 'optimizer_state_dict': optimizer.state_dict()}
			torch.save(state, os.path.join(
				'snapshots', f'large_scale_vrd-Epoch-{epoch}.pth'))
			print(f"Epoch {epoch} model saved!\n")


if __name__ == "__main__":
	main_worker()
