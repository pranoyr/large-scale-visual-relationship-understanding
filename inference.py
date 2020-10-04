

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

from torch.jit.annotations import Optional, List, Dict, Tuple
from torchvision.models.detection.faster_rcnn import MultiScaleRoIAlign, TwoMLPHead, FastRCNNPredictor 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


with open(os.path.join('/Users/pranoyr/code/Pytorch/faster-rcnn.pytorch/data/VRD', 'json_dataset', 'objects.json'), 'r') as f:
	objects = json.load(f)

classes = ['__background__']
classes.extend(objects)
num_classes = len(classes)
# self._classes.extend(self.predicates)
ind_to_class = dict(zip(range(num_classes), classes))


faster_rcnn = FasterRCNN().to(DEVICE)
faster_rcnn.eval()


# load pretrained weights
# checkpoint = torch.load('./snapshots/faster_rcnn_custom.pth', map_location='cpu')
checkpoint = torch.load('/Users/pranoyr/Downloads/faster_rcnn.pth', map_location='cpu')
faster_rcnn.load_state_dict(checkpoint['state_dict'], strict=True)
print("Model Restored")


im = Image.open('/Users/pranoyr/Downloads/bike.jpg')
img = np.array(im)
draw = img.copy()
# draw = cv2.resize(draw,(1344,768))
img = torch.from_numpy(img)
img = img.permute(2,0,1)
img = img.type(torch.float32)

detections, losses = faster_rcnn([img])
boxes = detections[0]['boxes']
scores = detections[0]['scores']
labels =  detections[0]['labels']
print(scores.shape)

print(boxes.shape)
for i in range(boxes.size(0)):
	box = boxes[i]
	score = scores[i]
	label = f"{ind_to_class[labels[i].item()]}: {scores[i].item():.2f}"
	cv2.rectangle(draw, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
	label = f"""{ind_to_class[labels[i].item()]}: {scores[i].item():.2f}"""
	label = f"{ind_to_class[labels[i].item()]}: {scores[i].item():.2f}"
	cv2.putText(draw, label,
				(box[0] + 20, box[1] + 40),
				cv2.FONT_HERSHEY_SIMPLEX,
				1,  # font scale
				(255, 0, 255),
				2)  # line type
path = "./results/faster_rcnn_sample.jpg"
cv2.imwrite(path, draw)


