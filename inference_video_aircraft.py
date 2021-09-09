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


# def draw_boxes(img, box, label, _ind_to_class):
# 	cv2.rectangle(img, (int(box[0]), int(box[1])),
# 				  (int(box[2]), int(box[3])), (0, 255, 0), 2)
# 	text = f"{_ind_to_class[int(label)]}"
# 	coord = (int(box[0])+3, int(box[1])+7+10)
# 	cv2.putText(img, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# 	cv2.rectangle(draw, box_coords[0], box_coords[1], (0, 0, 255), cv2.FILLED)
# 	cv2.putText(draw, text, (text_offset_x, text_offset_y-5), font,
# 				font_size, (255, 255, 255), lineThickness, cv2.LINE_AA)
# 	return img

# Save video
out = cv2.VideoWriter('/Users/pranoyr/Desktop/demo1.avi',
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


cam = cv2.VideoCapture(opt.video_path)
while True:
	ret, frame = cam.read()
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

	with torch.no_grad():
		detections, losses = faster_rcnn([im])

	sbj_boxes = detections[0]['sbj_boxes']
	obj_boxes = detections[0]['obj_boxes']
	sbj_labels = detections[0]['sbj_labels']
	obj_labels = detections[0]['obj_labels']
	pred_labels = detections[0]['predicates']
	boxes = detections[0]['boxes']
	labels = detections[0]['labels']
	# scores = detections[0]['scores']

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
		# if pred in ['attach to', 'arrive near']:
		pred = pred.replace('attach to', 'attached').replace(
				'arrive near', 'arrived').replace('on the left of', 'on left').replace('on the right of', 'on right').replace('in front of', 'in front')

		# color = list(np.random.random(size=3) * 256)
		# font = cv2.FONT_HERSHEY_SIMPLEX
		# lineThickness = 1
		# font_size = 0.5
		# write sbj and obj
		# centr_sub = (int(sbj_box[0].item()),
					#  int(sbj_box[1].item()))
		# centr_obj = (int((obj_box[0].item() + obj_box[2].item())/2),
		# 			int((obj_box[1].item() + obj_box[3].item())/2))
		sbj_box = (int(sbj_box[0].item()),
				   int(sbj_box[1].item()),
				   int(sbj_box[2].item()),
				   int(sbj_box[3].item()))
		set_text(draw_rlp, sbj + ' '+ pred, sbj_box)

		# set_text(draw_rlp, obj,centr_obj)
		# draw line conencting sbj and obj
		# cv2.line(draw_rlp, centr_sub, centr_obj, color, thickness=2)
		# predicate_point = (
		# 	int((centr_sub[0] + centr_obj[0])/2), int((centr_sub[1] + centr_obj[1])/2))
		# set_text(draw_rlp, pred, predicate_point)
	# path = f"./results/rel-{opt.image_path.split('/')[-1]}"
	# cv2.imwrite(path, draw_rlp)

	for bbox, label in zip(boxes, labels):
		if _ind_to_class[int(label)] in  ['person', 'catering truck']:
				continue
		sbj_box = (int(bbox[0].item()),
				   int(bbox[1].item()),
				   int(bbox[2].item()),
				   int(bbox[3].item()))
		set_text(draw_rlp, _ind_to_class[int(label)].replace("aeroplane","aircraft"), sbj_box)
	# path = f"./results/objs-{opt.image_path.split('/')[-1]}"
	# cv2.imwrite(path, draw_objects)
	cv2.write(cv2.resize(draw_rlp,(1280,720)))
