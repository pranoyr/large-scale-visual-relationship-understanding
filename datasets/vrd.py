import numpy as np
import cv2
from shapely.geometry import box
from shapely.ops import cascaded_union
from PIL import Image
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import json

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def make_image_list(dataset_path, type):
	imgs_list = []
	with open(os.path.join(dataset_path, 'json_dataset', f'annotations_{type}.json'), 'r') as f:
		annotations = json.load(f)
	sg_images = os.listdir(os.path.join(
		dataset_path, 'sg_dataset', f'sg_{type}_images'))

	annotations_copy = annotations.copy()
	for ann in annotations.items():
		if(not annotations[ann[0]] or ann[0] not in sg_images):
			annotations_copy.pop(ann[0])

	for ann in annotations_copy.items():
		imgs_list.append(ann[0])
	return imgs_list


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

		self.root = os.path.join(
			self.dataset_path, 'sg_dataset', f'sg_{self.image_set}_images')

		self.classes = self.all_objects.copy()
		self.preds = self.predicates.copy()
		self.classes.insert(0, '__background__')
		self.preds.insert(0, 'unknown')

		self._class_to_ind = dict(zip(self.classes, range(len(self.classes))))
		self._preds_to_ind = dict(
			zip(self.preds, range(len(self.preds))))
		self.imgs_list = make_image_list(self.dataset_path, self.image_set)

	def __len__(self):
		return len(self.imgs_list)

	def image_path_from_index(self, img_name):
		"""
		Construct an image path from the image's "index" identifier.
		"""
		image_path = os.path.join(self.dataset_path, 'sg_dataset', f'sg_{self.image_set}_images',
								  img_name)
		assert os.path.exists(image_path), \
			'Path does not exist: {}'.format(image_path)
		return image_path

	def load_pascal_annotation(self, index):
		"""
		Load image and bounding boxes info from XML file in the PASCAL VOC
		format.
		"""
		boxes = []
		labels = []
		preds = []
		# preds = []
		annotation = self.annotations[index]
		for spo in annotation:
			gt_sbj_label = spo['subject']['category']
			gt_sbj_bbox = spo['subject']['bbox']
			gt_obj_label = spo['object']['category']
			gt_obj_bbox = spo['object']['bbox']
			predicate = spo['predicate']

			# prepare bboxes for subject and object
			gt_sbj_bbox = y1y2x1x2_to_x1y1x2y2(gt_sbj_bbox)
			gt_obj_bbox = y1y2x1x2_to_x1y1x2y2(gt_obj_bbox)
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
		img_name = self.imgs_list[index]
		boxes, labels, preds = self.load_pascal_annotation(img_name)
		img_path = self.image_path_from_index(img_name)
		img = Image.open(img_path)
		img = torch.from_numpy(np.array(img))
		img = img.permute(2, 0, 1)
		img = img.type(torch.float32)

		assert len(boxes) == len(
			labels), "boxes and labels should be of equal length"

		return {'boxes': torch.tensor(boxes, dtype=torch.float32),
				'labels': torch.tensor(labels, dtype=torch.int64),
				'preds': torch.tensor(preds, dtype=torch.int64),
				'img': img
				}


def collater(data):
	imgs = [s['img'] for s in data]
	annotations = [{"boxes": s['boxes'].to(DEVICE)} for s in data]
	for i, s in enumerate(data):
		annotations[i]['labels'] = s['labels'].to(DEVICE)
	for i, s in enumerate(data):
    		annotations[i]['preds'] = s['preds'].to(DEVICE)
	return imgs, annotations
