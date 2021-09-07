import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms, utils

from config import cfg
from modelling.model import FasterRCNN
from opts import parse_opts


def draw_boxes(img, box, label, _ind_to_class):
	cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
	text =  f"{_ind_to_class[int(label)]}"
	coord = (int(box[0])+3, int(box[1])+7+10)
	cv2.putText(img, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
	return img
	
def set_text(draw, text, text_pos):
	font = cv2.FONT_HERSHEY_SIMPLEX
	lineThickness = 1
	font_size = 0.5
	# set some text
	# get the width and height of the text box
	(text_width, text_height) = cv2.getTextSize(text, font, font_size, lineThickness)[0]
	# set the text start position
	text_offset_x,text_offset_y = text_pos
	# make the coords of the box with a small padding of two pixels
	box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
	cv2.rectangle(draw, box_coords[0], box_coords[1], color, cv2.FILLED)

	cv2.putText(draw, text, text_pos, font, font_size,
				(255,255,255), lineThickness, cv2.LINE_AA)


opt = parse_opts()
with open(os.path.join(cfg.DATASET_DIR, 'json_dataset', 'objects.json'), 'r') as f:
	objects = json.load(f)
with open(os.path.join(cfg.DATASET_DIR, 'json_dataset', 'predicates.json'), 'r') as f:
	predicates = json.load(f)
with open(os.path.join(cfg.DATASET_DIR, 'json_dataset', 'objects.json'), 'r') as f:
	all_objects = json.load(f)

classes = all_objects.copy()
predicates.insert(0, 'unknown')
classes.insert(0, '__background__')
_class_to_ind = dict(zip(classes, range(len(classes))))
_ind_to_class = {v: k for k, v in _class_to_ind.items()}

cfg.DEVICE = "cpu"
faster_rcnn = FasterRCNN().to(cfg.DEVICE)

# load pretrained weights
checkpoint = torch.load(opt.weight_path, map_location='cpu')
faster_rcnn.load_state_dict(checkpoint['state_dict'])
print("Model Restored")
faster_rcnn.eval()

transform = transforms.Compose([
			transforms.ToTensor()])

im = Image.open(opt.image_path)
img = np.array(im)
draw = img.copy()
draw_rlp = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
draw_objects = draw_rlp.copy()
im = transform(im)

with torch.no_grad():
	detections, losses = faster_rcnn([im])

sbj_boxes = detections[0]['sbj_boxes']
obj_boxes = detections[0]['obj_boxes']
sbj_labels = detections[0]['sbj_labels']
obj_labels = detections[0]['obj_labels']
pred_labels = detections[0]['predicates']
# boxes = detections[0]['boxes']
# labels = detections[0]['labels']
# scores = detections[0]['scores']

for sbj_box, obj_box, sbj_label, obj_label, pred  \
		in zip(sbj_boxes, obj_boxes, sbj_labels, obj_labels, pred_labels):
	sbj = objects[sbj_label]
	obj = objects[obj_label]
	pred = predicates[pred]
	if obj != 'aeroplane':
		continue
	print(sbj, pred, obj)
	color = list(np.random.random(size=3) * 256)
	font = cv2.FONT_HERSHEY_SIMPLEX
	lineThickness = 1
	font_size = 0.5
	# write sbj and obj
	centr_sub = (int((sbj_box[0].item() + sbj_box[2].item())/2),
				 int((sbj_box[1].item() + sbj_box[3].item())/2))
	centr_obj = (int((obj_box[0].item() + obj_box[2].item())/2),
				 int((obj_box[1].item() + obj_box[3].item())/2))
	set_text(draw_rlp, sbj,centr_sub)
	set_text(draw_rlp, obj,centr_obj)
	# draw line conencting sbj and obj
	cv2.line(draw_rlp, centr_sub, centr_obj, color, thickness=2)
	predicate_point = (
		int((centr_sub[0] + centr_obj[0])/2), int((centr_sub[1] + centr_obj[1])/2))
	set_text(draw_rlp, pred, predicate_point)
path = f"./results/rel-{opt.image_path.split('/')[-1]}"
cv2.imwrite(path, draw_rlp)

# for bbox, label in zip(boxes, labels):
# 	draw_objects = draw_boxes(draw_objects, bbox, label, _ind_to_class)
# path = f"./results/objs-{opt.image_path.split('/')[-1]}"
# cv2.imwrite(path, draw_objects)