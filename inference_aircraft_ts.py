import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ts import display_ts
import torchvision
import time
from PIL import Image
from torchvision import transforms, utils

from config import cfg
from modelling.model import FasterRCNN
from opts import parse_opts

trackable_objects = ["aeroplane", "arrived", "attached"]

# Save video
out = cv2.VideoWriter('./demo1.avi',
								cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
											  (1280, 720))
										
def set_text(draw, text, sbj_box):
	font = cv2.FONT_HERSHEY_SIMPLEX
	lineThickness = 1
	font_size = 0.5
	# set some text
	# get the width and height of the text box
	(text_width, text_height) = cv2.getTextSize(
		text, font, font_size, lineThickness)[0]
	# set the text start position
	# if not isinstance(sbj_box, int):
	# 	text_offset_x, text_offset_y = int(sbj_box[0].item()), int(sbj_box[1].item())
	# else:
	text_offset_x, text_offset_y = int(sbj_box[0]), int(sbj_box[1])
	# make the coords of the box with a small padding of two pixels
	box_coords = ((text_offset_x, text_offset_y), (text_offset_x +
				  text_width + 2, text_offset_y - text_height - 10))
	cv2.rectangle(draw, box_coords[0], box_coords[1], (0, 0, 255), cv2.FILLED)
	cv2.putText(draw, text, (text_offset_x, text_offset_y-5), font,
				font_size, (255, 255, 255), lineThickness, cv2.LINE_AA)
	cv2.rectangle(draw, (sbj_box[0], sbj_box[1]),
				  (sbj_box[2], sbj_box[3]), (0, 0, 255))


def create_preds_dict(predictions):   # prediction = [(class,box), (class,box), ...]
	preds_dict = {}
	for pred in predictions:
		if pred[0] not in preds_dict.keys():
			preds_dict[pred[0]] = []

		preds_dict[pred[0]].append(pred[1])

	# preds_dict = {"arrive_near" : [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]}
	return preds_dict	


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

cfg.DEVICE = "cuda:0"
faster_rcnn = FasterRCNN().to(cfg.DEVICE)

# load pretrained weights
checkpoint = torch.load(opt.weight_path)
faster_rcnn.load_state_dict(checkpoint['state_dict'])
print("Model Restored")
faster_rcnn.eval()

transform = transforms.Compose([
	transforms.ToTensor()])


cam = cv2.VideoCapture(opt.video_path)
fps = cam.get(cv2.CAP_PROP_FPS)
print(f"FPS : {fps}")										  

frame_no = 0
while True:
	ret, frame = cam.read()
	frame_no += 1
	if not ret:
		break
	# opt.image_path = f"{opt.images_dir}/{img_name}"
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	im = Image.fromarray(frame)
	img = np.array(im)
	draw = img.copy()
	draw_rlp = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
	# draw_objects = draw_rlp.copy()
	im = transform(im)

	try:
		with torch.no_grad():
			detections, losses = faster_rcnn([im])
	except:
		out.write(cv2.resize(draw_rlp,(1280,720)))
		continue


	sbj_boxes = detections[0]['sbj_boxes']
	obj_boxes = detections[0]['obj_boxes']
	sbj_labels = detections[0]['sbj_labels']
	obj_labels = detections[0]['obj_labels']
	pred_labels = detections[0]['predicates']
	boxes = detections[0]['boxes']
	labels = detections[0]['labels']
	# scores = detections[0]['scores']

	predictions1 = []
	for sbj_box, obj_box, sbj_label, obj_label, pred  \
			in zip(sbj_boxes, obj_boxes, sbj_labels, obj_labels, pred_labels):
		sbj = objects[sbj_label]
		obj = objects[obj_label]
		pred = predicates[pred]
		if obj != 'aeroplane':
			continue
		print(sbj, pred, obj)
		if sbj not in  ['person', 'catering truck']:
			continue
		pred = pred.replace('attach to', 'attached').replace(
				'arrive near', 'arrived').replace('on the left of', 'on left').replace('on the right of', 'on right').replace('in front of', 'in front')


		sbj_box = (int(sbj_box[0].item()),
				   int(sbj_box[1].item()),
				   int(sbj_box[2].item()),
				   int(sbj_box[3].item()))
		set_text(draw_rlp, sbj + ' '+ pred, sbj_box)
		
		if pred in trackable_objects:
			predictions1.append((pred, sbj_box))


	predictions2 = []
	for bbox, label in zip(boxes, labels):
		if _ind_to_class[int(label)] in  ['person', 'catering truck']:
				continue
		sbj_box = (int(bbox[0].item()),
				   int(bbox[1].item()),
				   int(bbox[2].item()),
				   int(bbox[3].item()))
		set_text(draw_rlp, _ind_to_class[int(label)].replace("aeroplane","aircraft"), sbj_box)

		if _ind_to_class[int(label)] in trackable_objects:
			predictions2.append((_ind_to_class[int(label)], sbj_box))

	if predictions1:
		preds_dict1 = create_preds_dict.append(predictions1)
	else:
		preds_dict1 = {}
	if predictions2:
		preds_dict2 = create_preds_dict.append(predictions2)
	else:
		preds_dict2 = {}
	
	preds_dict = {**preds_dict1, **preds_dict2}
	
	if preds_dict:
		display_ts(preds_dict, frame_no, fps)

	out.write(cv2.resize(draw_rlp,(1280,720)))
