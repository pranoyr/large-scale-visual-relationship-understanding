

from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead, AnchorGenerator
# from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 
# from datasets.pascal_voc import VOCDataset, collater
from datasets.vrd import VRDDataset, collater
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models.detection._utils as  det_utils
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


objects.insert(0,'__background__')
predicates.insert(0,'unknown')
print(predicates)

# classes = ['__background__']
# classes.extend(objects)
# num_classes = len(classes)
# # self._classes.extend(self.predicates)
# ind_to_class = dict(zip(range(num_classes), classes))


faster_rcnn = FasterRCNN().to(DEVICE)

# load pretrained weights
# checkpoint = torch.load('./snapshots/faster_rcnn_custom.pth', map_location='cpu')
checkpoint = torch.load('/Users/pranoyr/Downloads/faster_rcnn.pth', map_location='cpu')
faster_rcnn.load_state_dict(checkpoint['state_dict'])
print("Model Restored")

faster_rcnn.eval()





im = Image.open('/Users/pranoyr/Downloads/vrd_sample/12239689_0ad9e20e3a_b.jpg')
img = np.array(im)
draw = img.copy()
# draw = cv2.resize(draw,(1344,768))
img = torch.from_numpy(img)
img = img.permute(2,0,1)
img = img.type(torch.float32)


with torch.no_grad():
	detections, losses = faster_rcnn([img])

sbj_boxes = detections[0]['sbj_boxes']
obj_boxes = detections[0]['obj_boxes']
sbj_labels = detections[0]['sbj_labels']
obj_labels = detections[0]['obj_labels']
predicates = detections[0]['predicates']


for sbj_box, obj_box, sbj_label, obj_label, predicate  \
		in zip(sbj_boxes, obj_boxes, sbj_labels, obj_labels, predicates):

	# score = scores[i]
	# label = f"{ind_to_class[labels[i].item()]}: {scores[i].item():.2f}"
	# cv2.rectangle(draw, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
	# label = f"""{ind_to_class[labels[i].item()]}: {scores[i].item():.2f}"""
	# label = f"{ind_to_class[labels[i].item()]}: {scores[i].item():.2f}"

	label = f"{objects[sbj_label]}"
	cv2.rectangle(draw, (sbj_box[0], sbj_box[1]), (sbj_box[2], sbj_box[3]), (255, 255, 0), 4)
	cv2.putText(draw, label,
				(sbj_box[0] + 20, sbj_box[1] + 40),
				cv2.FONT_HERSHEY_SIMPLEX,
				1,  # font scale
				(255, 0, 255),
				2)  # line type
	
	label = f"{objects[obj_label]}"
	cv2.rectangle(draw, (obj_box[0], obj_box[1]), (obj_box[2], obj_box[3]), (255, 255, 0), 4)
	cv2.putText(draw, label,
				(obj_box[0] + 20, obj_box[1] + 40),
				cv2.FONT_HERSHEY_SIMPLEX,
				1,  # font scale
				(255, 0, 255),
				2)  # line type


path = "./results/faster_rcnn_sample.jpg"
cv2.imwrite(path, draw)


