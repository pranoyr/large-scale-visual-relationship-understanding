
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

from config import cfg
from modelling.model import FasterRCNN
from opts import parse_opts

opt = parse_opts()

with open(os.path.join(cfg.DATASET_DIR, 'json_dataset', 'objects.json'), 'r') as f:
    objects = json.load(f)
with open(os.path.join(cfg.DATASET_DIR, 'json_dataset', 'predicates.json'), 'r') as f:
    predicates = json.load(f)
predicates.insert(0, 'unknown')

faster_rcnn = FasterRCNN().to(cfg.DEVICE)

# load pretrained weights
checkpoint = torch.load(opt.weight_path, map_location='cpu')
faster_rcnn.load_state_dict(checkpoint['state_dict'])
print("Model Restored")
faster_rcnn.eval()

im = Image.open(opt.image_path)
img = np.array(im)
draw = img.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
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

    sbj = objects[sbj_label]
    obj = objects[obj_label]
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
