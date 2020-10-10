from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
import os
import os.path as osp
from ast import literal_eval

import numpy as np
import six
import torch
import torch.nn as nn
import yaml
from packaging import version
from torch.nn import init

from utils.collections import AttrDict

__C = AttrDict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

# Training options
__C.N_EPOCHS = 100
__C.BATCH_SIZE = 4
__C.LR_RATE = 1e-5
__C.WORKERS = 0
__C.WEIGHT_DECAY = 1e-4

# Box parameters
__C.SCORE_THRESH = 0.5
__C.NMS_THRESH = 0.5
__C.BOX_DETECTIONS_PER_IMG = 100
__C.FG_IOU_THRESH = 0.5
__C.BG_IOU_THRESH = 0.5
__C.BATCH_SIZE_PER_IMAGE = 512
__C.POSITIVE_FRACTION = 0.25
__C.NUM_CLASSES = 101

# Subject/Object Branch Paramaters
__C.BATCH_SIZE_PER_IMAGE_SO = 64
__C.POSITIVE_FRACTION_SO = 0.5
# Relationship Branch Parameters
__C.BATCH_SIZE_PER_IMAGE_REL = 128
__C.POSITIVE_FRACTION_REL = 0.25
__C.NORM_SCALE = 5.0


# RPN parameters,
__C.RPN_PRE_NMS_TOP_N_TRAIN = 2000
__C.RPN_PRE_NMS_TOP_N_TEST = 1000
__C.RPN_POST_NMS_TOP_N_TRAIN = 2000
__C.RPN_POST_NMS_TOP_N_TEST = 1000
__C.RPN_NMS_THRESH = 0.7
__C.RPN_FG_IOU_THRESH = 0.7
__C.RPN_BG_IOU_THRESH = 0.3
__C.RPN_BATCH_SIZE_PER_IMAGE = 256
__C.RPN_POSITIVE_FRACTION = 0.5

__C.DEVICE = 'cpu'

# Data directory
__C.DATASET_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'data', 'VRD')
__C.WORD_VECTORS_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'data', 'wordvectors', 'GoogleNews-vectors-negative300.bin')
