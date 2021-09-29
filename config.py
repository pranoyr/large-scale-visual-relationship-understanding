from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
from utils.collections import AttrDict

__C = AttrDict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C.DEVICE = 'cuda'

#
# Training options
#
__C.TRAIN = AttrDict()

# Optmization Algorithm
__C.TRAIN.TYPE = "SGD"

# Base learning rate for the specified schedule
__C.TRAIN.LEARNING_RATE = 0.0001

__C.TRAIN.BACKBONE_LR_SCALAR = 0.1

# Schedule type (see functions in utils.lr_policy for options)
# E.g., 'step', 'steps_with_decay', ...
__C.TRAIN.LR_POLICY = 'step'

__C.TRAIN.GAMMA = 0.1

# Uniform step size for 'steps' policy
__C.TRAIN.STEP_SIZE = 30000

# Non-uniform step iterations for 'steps_with_decay' or 'steps_with_lrs'
# policies
__C.TRAIN.STEPS = []

# Learning rates to use with 'steps_with_lrs' policy
__C.TRAIN.LRS = []

# Maximum number of SGD iterations
__C.TRAIN.MAX_ITER = 40000

# Momentum to use with SGD
__C.TRAIN.MOMENTUM = 0.9

# L2 regularization hyperparameter
__C.TRAIN.WEIGHT_DECAY = 0.0005

# Whether to double the learning rate for bias
__C.TRAIN.DOUBLE_BIAS = True

# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False

# Warm up to TRAIN.BASE_LR over this number of SGD iterations
__C.TRAIN.WARM_UP_ITERS = 500

# Start the warm up from TRAIN.BASE_LR * TRAIN.WARM_UP_FACTOR
__C.TRAIN.WARM_UP_FACTOR = 1.0 / 3.0

# WARM_UP_METHOD can be either 'constant' or 'linear' (i.e., gradual)
__C.TRAIN.WARM_UP_METHOD = 'linear'

# Scale the momentum update history by new_lr / old_lr when updating the
# learning rate (this is correct given MomentumSGDUpdateOp)
__C.TRAIN.SCALE_MOMENTUM = False
# Only apply the correction if the relative LR change exceeds this threshold
# (prevents ever change in linear warm up from scaling the momentum by a tiny
# amount; momentum scaling is only important if the LR change is large)
__C.TRAIN.SCALE_MOMENTUM_THRESHOLD = 1.1

# Suppress logging of changes to LR unless the relative change exceeds this
# threshold (prevents linear warm up from spamming the training log)
__C.TRAIN.LOG_LR_CHANGE_THRESHOLD = 1.1


#
# Box Paramters
#

__C.BOX = AttrDict()

__C.BOX.SCORE_THRESH = 0.6

__C.BOX.NMS_THRESH = 0.4

__C.BOX.DETECTIONS_PER_IMG = 100

__C.BOX.FG_IOU_THRESH = 0.5

__C.BOX.BG_IOU_THRESH = 0.5

__C.BOX.BATCH_SIZE_PER_IMAGE = 512

__C.BOX.POSITIVE_FRACTION = 0.25

__C.BOX.NUM_CLASSES = 9


#
# Subject, Object, Relation Branch Paramaters
#

__C.MODEL = AttrDict()

__C.MODEL.BATCH_SIZE_PER_IMAGE_SO = 64

__C.MODEL.POSITIVE_FRACTION_SO = 0.5

__C.MODEL.BATCH_SIZE_PER_IMAGE_REL = 128

__C.MODEL.POSITIVE_FRACTION_REL = 0.5

__C.MODEL.NORM_SCALE = 10.0

__C.MODEL.USE_SEM_CONCAT = False


#
# RPN Paramaters
#

__C.RPN = AttrDict()

__C.RPN.PRE_NMS_TOP_N_TRAIN = 2000

__C.RPN.PRE_NMS_TOP_N_TEST = 1000

__C.RPN.POST_NMS_TOP_N_TRAIN = 2000

__C.RPN.POST_NMS_TOP_N_TEST = 1000

__C.RPN.NMS_THRESH = 0.7

__C.RPN.FG_IOU_THRESH = 0.7

__C.RPN.BG_IOU_THRESH = 0.3

__C.RPN.BATCH_SIZE_PER_IMAGE = 256

__C.RPN.POSITIVE_FRACTION = 0.5


#
# Test Paramaters
#

__C.TEST = AttrDict()
__C.TEST.THRESHOLD = 0.0


#
# Dataset, Word Vectors Directory
#
__C.DATASET = "Aircraft"
__C.DATASET_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'data', __C.DATASET)

__C.WORD_VECTORS_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'data', 'wordvectors', 'GoogleNews-vectors-negative300.bin')
