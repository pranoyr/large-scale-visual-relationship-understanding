from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import os
import os.path as osp
import copy
from ast import literal_eval

import numpy as np
from packaging import version
import torch
import torch.nn as nn
from torch.nn import init
import yaml

from utils.collections import AttrDict

__C = AttrDict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()

# Datasets to train on
# Available dataset list: datasets.dataset_catalog.DATASETS.keys()
# If multiple datasets are listed, the model is trained on their union
__C.TRAIN.DATASETS = ()

__C.N_EPOCHS = 100

# Box parameters
__C.SCORE_THRESH = 0.5
__C.NMS_THRESH = 0.5
__C.BOX_DETECTIONS_PER_IMG = 100
__C.FG_IOU_THRESH = 0.5
__C.BG_IOU_THRESH = 0.5
__C.BATCH_SIZE_PER_IMAGE = 512
__C.POSITIVE_FRACTION = 0.25
__C.NUM_CLASSES = 101

__C.BATCH_SIZE_PER_IMAGE_SO = 64
__C.POSITIVE_FRACTION_SO = 0.5
__C.BATCH_SIZE_PER_IMAGE_REL = 128,
__C.POSITIVE_FRACTION_REL = 0.5


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


# ------------------------------

# Data directory
__C.DATASET_DIR = '/Users/pranoyr/code/Pytorch/faster-rcnn.pytorch/data/VRD'
__C.WORD_VECTORS_DIR = '/Users/pranoyr/Downloads/GoogleNews-vectors-negative300.bin'

# # Data directory
# __C.DATASET_DIR = '/home/neuroplex/code/faster-rcnn/data/VRD'
# __C.WORD_VECTORS_DIR = '/home/neuroplex/data/GoogleNews-vectors-negative300.bin'
