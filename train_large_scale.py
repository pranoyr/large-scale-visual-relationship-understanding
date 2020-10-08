

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
    dataloader = DataLoader(
        dataset_train, num_workers=cfg.WORKERS, collate_fn=collater, batch_size=cfg.BATCH_SIZE)

    faster_rcnn = FasterRCNN().to(cfg.DEVICE)
    optimizer = optim.Adam(faster_rcnn.parameters(), lr=cfg.LR_RATE)

    # resume model
    if opt.weight_path:
        resume_model(opt, faster_rcnn, optimizer)

    faster_rcnn.train()
    for epoch in range(1, cfg.N_EPOCHS+1):
        for i, data in enumerate(dataloader):
            images, targets = data
            _, losses = faster_rcnn(images, targets)
            final_loss = losses["loss_objectness"] + losses["loss_rpn_box_reg"] + \
                losses["loss_classifier"] + losses["loss_box_reg"] + \
                losses["loss_sbj"] + losses["loss_obj"] + losses["loss_rlp"]

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"""RCNN_Loss    : {final_loss.item()}
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

        state = {'epoch': epoch, 'state_dict': faster_rcnn.state_dict(
        ), 'optimizer_state_dict': optimizer.state_dict()}
        torch.save(state, os.path.join(
            'snapshots', f'large_scale_vrd-Epoch-{epoch}.pth'))
        print(f"Epoch {epoch} model saved!\n")


if __name__ == "__main__":
    main_worker()
