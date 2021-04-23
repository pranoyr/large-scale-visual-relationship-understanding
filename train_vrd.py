

import math
import os
import pdb
import random
import time
from collections import OrderedDict
import utils.net as net_utils

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
from dataset import get_training_data, get_validation_data

from config import cfg
from datasets.vrd import collater
from modelling.model import FasterRCNN
from opts import parse_opts
from utils.util import AverageMeter, Metrics, ProgressMeter


def val_epoch(model, dataloader):
    losses_sbj = AverageMeter('Loss', ':.4e')
    losses_obj = AverageMeter('Loss', ':.4e')
    losses_rel = AverageMeter('Loss', ':.4e')
    losses_total = AverageMeter('Loss', ':.4e')

    for i, data in enumerate(dataloader):
        images, targets = data
        with torch.no_grad():
            _, metrics = model(images, targets)
        final_loss = metrics["loss_objectness"] + metrics["loss_rpn_box_reg"] + \
            metrics["loss_classifier"] + metrics["loss_box_reg"] + \
            metrics["loss_sbj"] + metrics["loss_obj"] + metrics["loss_rlp"]

        losses_sbj.update(metrics["loss_sbj"].item(), len(images))
        losses_obj.update(metrics["loss_obj"].item(), len(images))
        losses_rel.update(metrics["loss_rlp"].item(), len(images))
        losses_total.update(final_loss.item(), len(images))

    losses = {}
    losses['total_loss'] = losses_total.avg
    losses['sbj_loss'] = losses_sbj.avg
    losses['obj_loss'] = losses_obj.avg
    losses['rel_loss'] = losses_rel.avg
    return losses


def load_from_ckpt(opt, model):
    """ Loading model from checkpoint.
    """
    checkpoint = torch.load(opt.weight_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("** Loaded model **")


def load_train_utils(opt, optimizer, scheduler):
    """ loading optmizer, scheduler.
    """
    checkpoint = torch.load(opt.weight_path)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    step = checkpoint['step'] + 1
    print(" ** Loaded optmizer and scheduler ** ")
    return step


def save_model(model, optimizer, scheduler, step):
    """ Saving model and train_utils.
    """
    state = {'step': step, 'state_dict': model.state_dict(
    ), 'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(state, os.path.join(
        'snapshots', f'large_scale_vrd_model.pth'))


def main_worker():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    opt = parse_opts()
    train_data = get_training_data(cfg)
    val_data = get_validation_data(cfg)
    train_loader = DataLoader(
        train_data, num_workers=opt.num_workers, collate_fn=collater, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_data, num_workers=opt.num_workers, collate_fn=collater, batch_size=opt.batch_size, shuffle=True)

    print(f"Training dataset size : {len(train_loader.dataset)}")
    print(f"Validation dataset size : {len(val_loader.dataset)}")

    dataiterator = iter(train_loader)

    faster_rcnn = FasterRCNN()

    # loading model from a ckpt
    if opt.weight_path:
        load_from_ckpt(opt, faster_rcnn)
    faster_rcnn.to(cfg.DEVICE)

    lr = cfg.TRAIN.LEARNING_RATE
    
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

    # scheduler
    if opt.scheduler == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5)
    elif opt.scheduler == "multi_step":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=[83631, 111508])
    elif opt.scheduler == "step_lr":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.1, last_epoch=-1)

    if opt.weight_path:
        opt.begin_iter = load_train_utils(opt, optimizer, scheduler)

    # lr of non-backbone parameters, for commmand line outputs.
    lr = optimizer.param_groups[2]['lr']
    # lr of backbone parameters, for commmand line outputs.
    # backbone_lr = optimizer.param_groups[0]['lr']

    summary_writer = Metrics(log_dir='tf_logs')

    losses_sbj = AverageMeter('Sbj loss: ', ':.2f')
    losses_obj = AverageMeter('Obj loss: ', ':.2f')
    losses_rel = AverageMeter('Rel loss: ', ':.2f')
    losses_total = AverageMeter('Total loss: ', ':.2f')
    progress = ProgressMeter([losses_sbj, losses_obj, losses_rel, losses_total],
                             prefix='Train: ')

    faster_rcnn.train()
    th = 10000
    for step in range(opt.begin_iter, opt.max_iter):
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

        losses_sbj.update(metrics["loss_sbj"].item(), len(images))
        losses_obj.update(metrics["loss_obj"].item(), len(images))
        losses_rel.update(metrics["loss_rlp"].item(), len(images))
        losses_total.update(final_loss.item(), len(images))

        if (step) % 10 == 0:
            progress.display(step)

        if step % 2500 == 0:
            train_losses = {}
            train_losses['total_loss'] = losses_total.avg
            train_losses['sbj_loss'] = losses_sbj.avg
            train_losses['obj_loss'] = losses_obj.avg
            train_losses['rel_loss'] = losses_rel.avg
            val_losses = val_epoch(faster_rcnn, val_loader)

            if opt.scheduler == "plateau":
                scheduler.step(val_losses['total_loss'])
            lr = optimizer.param_groups[0]['lr']

            if val_losses['total_loss'] < th:
                save_model(faster_rcnn, optimizer, scheduler, step)
                print(f"*** Saved model ***")
                th = val_losses['total_loss']

             # write summary
            summary_writer.log_metrics(train_losses, val_losses, step, lr)

            print(
                f"* Average training loss : {train_losses['total_loss']:.3f}")
            print(
                f"* Average validation loss : {val_losses['total_loss']:.3f}")

            losses_sbj.reset()
            losses_obj.reset()
            losses_rel.reset()
            losses_total.reset()


if __name__ == "__main__":
    main_worker()
