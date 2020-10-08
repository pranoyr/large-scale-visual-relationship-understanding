

from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead, AnchorGenerator
# from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# from datasets.pascal_voc import VOCDataset, collater
from datasets.vrd import VRDDataset, collater
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models.detection._utils as det_utils
from torchvision.ops import boxes as box_ops
from PIL import Image
import json
import os

from collections import OrderedDict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
import time
import pdb
from torchvision.models.resnet import resnet101
import torchvision
import math
import cv2
from modelling.model import FasterRCNN
from config import cfg

from torch.jit.annotations import Optional, List, Dict, Tuple
from torchvision.models.detection.faster_rcnn import MultiScaleRoIAlign, TwoMLPHead, FastRCNNPredictor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


with open(os.path.join(cfg.DATASET_DIR, 'json_dataset', 'objects.json'), 'r') as f:
	objects = json.load(f)

with open(os.path.join(cfg.DATASET_DIR, 'json_dataset', 'predicates.json'), 'r') as f:
	predicates = json.load(f)


predicates.insert(0, 'unknown')
print(predicates)

faster_rcnn = FasterRCNN().to(DEVICE)

# load pretrained weights
# checkpoint = torch.load('./snapshots/faster_rcnn_custom.pth', map_location='cpu')
checkpoint = torch.load(
	'/Users/pranoyr/Downloads/faster_rcnn.pth', map_location='cpu')
faster_rcnn.load_state_dict(checkpoint['state_dict'])
print("Model Restored")

faster_rcnn.eval()

im = Image.open('/Users/pranoyr/Downloads/IMG_8487.jpg')
img = np.array(im)
draw = img.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
# draw = cv2.resize(draw,(1344,768))
img = torch.from_numpy(img)
img = img.permute(2, 0, 1)
img = img.type(torch.float32)


with torch.no_grad():
	detections, losses = faster_rcnn([img])

sbj_boxes = detections[0]['sbj_boxes']
obj_boxes = detections[0]['obj_boxes']
sbj_labels = detections[0]['sbj_labels']
obj_labels = detections[0]['obj_labels']
pred_labels = detections[0]['predicates']


for sbj_box, obj_box, sbj_label, obj_label, pred  \
		in zip(sbj_boxes, obj_boxes, sbj_labels, obj_labels, pred_labels):


	sbj = f"{objects[sbj_label]}"
	obj = f"{objects[obj_label]}"

	pred = predicates[pred]
	print(sbj, pred, obj)

	font = cv2.FONT_HERSHEY_SIMPLEX
	lineThickness = 2
	font_size = 1

	# write sbj and obj
	centr_sub = (int((sbj_box[0].item() + sbj_box[2].item())/2),
				 int((sbj_box[1].item() + sbj_box[3].item())/2))
	centr_obj = (int((obj_box[0].item() + obj_box[2].item())/2),
				 int((obj_box[1].item() + obj_box[3].item())/2))
	cv2.putText(draw, sbj, centr_sub, font, font_size,
				(0, 0, 255), lineThickness, cv2.LINE_AA)
	cv2.putText(draw, obj, centr_obj, font, font_size,
				(0, 0, 255), lineThickness, cv2.LINE_AA)

	# draw line conencting sbj and obj
	cv2.line(draw, centr_sub, centr_obj, (0, 255, 0), lineThickness)
	predicate_point = (
		int((centr_sub[0] + centr_obj[0])/2), int((centr_sub[1] + centr_obj[1])/2))
	# write predicate
	cv2.putText(draw, pred, predicate_point, font, font_size,
				(0, 0, 255), lineThickness, cv2.LINE_AA)


path = "./results/faster_rcnn_sample.jpg"
cv2.imwrite(path, draw)
