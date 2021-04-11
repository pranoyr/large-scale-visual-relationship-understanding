import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from config import cfg
import ast
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils


def y1y2x1x2_to_x1y1x2y2(y1y2x1x2):
	x1 = y1y2x1x2[2]
	y1 = y1y2x1x2[0]
	x2 = y1y2x1x2[3]
	y2 = y1y2x1x2[1]
	return [x1, y1, x2, y2]


def one_hot_encode(integer_encoding, num_classes):
	""" One hot encode.
	"""
	onehot_encoded = [0 for _ in range(num_classes)]
	onehot_encoded[integer_encoding] = 1
	return onehot_encoded


class VRDDataset(Dataset):
	"""VRD dataset."""

	def __init__(self, dataset_path, image_set):
		self.dataset_path = dataset_path
		self.image_set = image_set
		# read annotations file
		with open(os.path.join(self.dataset_path, 'json_dataset', f'annotations_{self.image_set}.json'), 'r') as f:
			self.annotations = json.load(f)
		with open(os.path.join(self.dataset_path, 'json_dataset', 'objects.json'), 'r') as f:
			self.all_objects = json.load(f)
		with open(os.path.join(self.dataset_path, 'json_dataset', 'predicates.json'), 'r') as f:
			self.predicates = json.load(f)


		self.classes = self.all_objects.copy()
		self.preds = self.predicates.copy()
		self.classes.insert(0, '__background__')
		self.preds.insert(0, 'unknown')

		self._class_to_ind = dict(zip(self.classes, range(len(self.classes))))
		self._preds_to_ind = dict(
			zip(self.preds, range(len(self.preds))))

		with open(os.path.join(self.dataset_path, 'ImageSets', image_set +'.txt'), 'r') as file:
			self.ids_list = file.readlines()

		self.transform = transforms.Compose([
			transforms.ToTensor()])

	def __len__(self):
		return len(self.ids_list)

	def get_image(self, filename):
		"""
		Construct an image path from the image's "index" identifier.
		"""
		image_path = os.path.join(self.dataset_path, 'JPEGImages', filename, '.jpg')
		img = Image.open(image_path)
		return img

	def load_annotations(self, filename):
		"""
		Load image and bounding boxes info from XML file in the PASCAL VOC
		format.
		"""

		annotations = pd.read_csv(f'{self.dataset_path}/vrd/{filename}.csv')
		annotations = annotations.values.tolist()
		boxes = []
		labels = []
		preds = []
		for spo in annotations:
			gt_sbj_label, gt_obj_label = spo[0], spo[3]
			gt_sbj_bbox, gt_obj_bbox = ast.literal_eval(spo[1]), ast.literal_eval(spo[4])
			predicate = spo[2]

			# prepare bboxes for subject and object
			boxes.append([gt_sbj_bbox, gt_obj_bbox])

			# prepare labels for subject and object
			# map to word
			gt_sbj_label = self.all_objects[gt_sbj_label]
			gt_obj_label = self.all_objects[gt_obj_label]
			predicate = self.predicates[predicate]
			# map to new index
			labels.append([self._class_to_ind[gt_sbj_label],
						   self._class_to_ind[gt_obj_label]])
			preds.append(self._preds_to_ind[predicate])
		return boxes, labels, preds

	def __getitem__(self, index):
		filename = self.ids_list[index]
		boxes, labels, preds = self.load_annotations(filename)
		img = self.get_image(filename)
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


if __name__=='__main__':
	dataset = pd.read_csv("/Users/pranoyr/Downloads/Visual-Relationshiop-Detection-Annotation-tool-master/demo/vrd/test.csv")
	dataset = dataset.values.tolist()
	res = ast.literal_eval(dataset[0][1])


