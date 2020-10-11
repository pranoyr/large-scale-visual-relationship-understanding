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

__C.SOLVER = AttrDict()

# Training options
__C.SOLVER.N_EPOCHS = 100
__C.SOLVER.BATCH_SIZE = 2
__C.SOLVER.WORKERS = 0

# e.g 'SGD', 'Adam'
__C.SOLVER.TYPE = 'Adam'

# Base learning rate for the specified schedule
__C.SOLVER.BASE_LR = 0.001

__C.SOLVER.BACKBONE_LR_SCALAR = 0.1

# Schedule type (see functions in utils.lr_policy for options)
# E.g., 'step', 'steps_with_decay', ...
__C.SOLVER.LR_POLICY = 'step'

# Some LR Policies (by example):
# 'step'
#   lr = SOLVER.BASE_LR * SOLVER.GAMMA ** (cur_iter // SOLVER.STEP_SIZE)
# 'steps_with_decay'
#   SOLVER.STEPS = [0, 60000, 80000]
#   SOLVER.GAMMA = 0.1
#   lr = SOLVER.BASE_LR * SOLVER.GAMMA ** current_step
#   iters [0, 59999] are in current_step = 0, iters [60000, 79999] are in
#   current_step = 1, and so on
# 'steps_with_lrs'
#   SOLVER.STEPS = [0, 60000, 80000]
#   SOLVER.LRS = [0.02, 0.002, 0.0002]
#   lr = LRS[current_step]

# Hyperparameter used by the specified policy
# For 'step', the current LR is multiplied by SOLVER.GAMMA at each step
__C.SOLVER.GAMMA = 0.1

# Uniform step size for 'steps' policy
__C.SOLVER.STEP_SIZE = 30000

# Non-uniform step iterations for 'steps_with_decay' or 'steps_with_lrs'
# policies
__C.SOLVER.STEPS = []

# Learning rates to use with 'steps_with_lrs' policy
__C.SOLVER.LRS = []

# Maximum number of SGD iterations
__C.SOLVER.MAX_ITER = 40000

# Momentum to use with SGD
__C.SOLVER.MOMENTUM = 0.9

# L2 regularization hyperparameter
__C.SOLVER.WEIGHT_DECAY = 0.0005
# L2 regularization hyperparameter for GroupNorm's parameters
__C.SOLVER.WEIGHT_DECAY_GN = 0.0

# Whether to double the learning rate for bias
__C.SOLVER.BIAS_DOUBLE_LR = True

# Whether to have weight decay on bias as well
__C.SOLVER.BIAS_WEIGHT_DECAY = False

# Warm up to SOLVER.BASE_LR over this number of SGD iterations
__C.SOLVER.WARM_UP_ITERS = 500

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARM_UP_FACTOR
__C.SOLVER.WARM_UP_FACTOR = 1.0 / 3.0

# WARM_UP_METHOD can be either 'constafnt' or 'linear' (i.e., gradual)
__C.SOLVER.WARM_UP_METHOD = 'linear'

# Scale the momentum update history by new_lr / old_lr when updating the
# learning rate (this is correct given MomentumSGDUpdateOp)
__C.SOLVER.SCALE_MOMENTUM = True

# Only apply the correction if the relative LR change exceeds this threshold
# (prevents ever change in linear warm up from scaling the momentum by a tiny
# amount; momentum scaling is only important if the LR change is large)
__C.SOLVER.SCALE_MOMENTUM_THRESHOLD = 1.1

# Suppress logging of changes to LR unless the relative change exceeds this
# threshold (prevents linear warm up from spamming the training log)
__C.SOLVER.LOG_LR_CHANGE_THRESHOLD = 1.1



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
__C.NORM_SCALE = 3.0


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

__C.DEVICE = 'cuda'

# Data directory
__C.DATASET_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'data', 'VRD')
__C.WORD_VECTORS_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'data', 'wordvectors', 'GoogleNews-vectors-negative300.bin')
