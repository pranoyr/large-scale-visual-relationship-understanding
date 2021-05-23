

import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from config import cfg
from PIL import Image
from shapely.geometry import box
from shapely.ops import cascaded_union
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from utils.boxes import xywh_to_xyxy


class VGDataset(Dataset):
	"""Visual Genome dataset."""

	def __init__(self, dataset_path, image_set):
		self.dataset_path = dataset_path
		self.image_set = image_set
		# read annotations file
		with open(os.path.join(self.dataset_path, 'json_dataset', 'relationships.json'), 'r') as f:
			self.data = json.load(f)
		with open(os.path.join(self.dataset_path, 'json_dataset', 'objects.json'), 'r') as f:
			all_objects = json.load(f)
		with open(os.path.join(self.dataset_path, 'json_dataset', 'predicates.json'), 'r') as f:
			predicates = json.load(f)

		self.root = os.path.join(
			self.dataset_path, 'images')

		self.classes = all_objects.copy()
		self.preds = predicates.copy()
		self.classes.insert(0, '__background__')
		print(f"Total object classes {len(self.classes)}")
		self.preds.insert(0, 'unknown')

		self._class_to_ind = dict(zip(self.classes, range(len(self.classes))))
		self._preds_to_ind = dict(
			zip(self.preds, range(len(self.preds))))

		self.transform = transforms.Compose([
			transforms.ToTensor()])

	def __len__(self):
		return len(self.data)

	def image_path_from_id(self, img_id):
		"""
		Construct an image path from the image's "index" identifier.
		"""
		image_path = os.path.join(self.dataset_path, 'images', img_id, '.jpg')
		assert os.path.exists(image_path), \
			'Path does not exist: {}'.format(image_path)
		return image_path

	def load_annotation(self, index):
		"""
		Load image and bounding boxes info from XML file in the PASCAL VOC
		format.
		"""
		boxes = []
		labels = []
		preds = []
		for spo in self.data[index]['relationships']:
			try:
				gt_sbj_label = spo['subject']['name']
			except:
				gt_sbj_label = ''.join(spo['subject']['names'][0])
			gt_sbj_bbox = spo['subject']['x'], spo['subject']['y'], spo['subject']['w'], spo['subject']['h']

			try:
				gt_obj_label = spo['object']['name']
			except:
				gt_obj_label = ''.join(spo['object']['names'][0])
			gt_obj_bbox = spo['object']['x'], spo['object']['y'], spo['object']['w'], spo['object']['h']

			predicate = spo['predicate']

			if (gt_sbj_label not in self.classes or gt_obj_label not in self.classes or predicate not in self.preds):
				continue

			# prepare bboxes for subject and object
			gt_sbj_bbox = xywh_to_xyxy(gt_sbj_bbox)
			gt_obj_bbox = xywh_to_xyxy(gt_obj_bbox)
			boxes.append([gt_sbj_bbox, gt_obj_bbox])

			# prepare labels for subject and object
			# map to index
			labels.append([self._class_to_ind[gt_sbj_label],
						   self._class_to_ind[gt_obj_label]])
			preds.append(self._preds_to_ind[predicate])
		return boxes, labels, preds

	def __getitem__(self, index):
		img_id = self.data[index]['image_id']
		img_path = self.image_path_from_id(img_id)
		img = Image.open(img_path)
		boxes, labels, preds = self.load_annotation(index)
		img = self.transform(img)

		assert len(boxes) == len(
			labels), "boxes and labels should be of equal length"

		return {'boxes': torch.tensor(boxes, dtype=torch.float32),
				'labels': torch.tensor(labels, dtype=torch.int64),
				'preds': torch.tensor(preds, dtype=torch.int64),
				'img': img
				}


def collater(data):
	imgs = [s['img'] for s in data]
	annotations = [{"boxes": s['boxes'].to(cfg.DEVICE)} for s in data]
	for i, s in enumerate(data):
		annotations[i]['labels'] = s['labels'].to(cfg.DEVICE)
	for i, s in enumerate(data):
		annotations[i]['preds'] = s['preds'].to(cfg.DEVICE)
	return imgs, annotations
