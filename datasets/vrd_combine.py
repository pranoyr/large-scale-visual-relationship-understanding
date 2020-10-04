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
			self.objects = json.load(f)
		self.root = os.path.join(
			self.dataset_path, 'sg_dataset', f'sg_{self.image_set}_images')


		self.classes = ['__background__']
		self.classes.extend(self.objects)
		num_classes = len(self.classes)
		# self._classes.extend(self.predicates)
		self._class_to_ind = dict(zip(self.classes, range(num_classes)))
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
		# preds = []
		annotation = self.annotations[index]
		for sub_pred_obj in annotation:
			gt_subject_label = sub_pred_obj['subject']['category']
			gt_subject_bbox = sub_pred_obj['subject']['bbox']
			gt_object_label = sub_pred_obj['object']['category']
			gt_object_bbox = sub_pred_obj['object']['bbox']
			
			# prepare bboxes for subject and object 
			gt_subject_bbox = y1y2x1x2_to_x1y1x2y2(gt_subject_bbox)
			gt_subject_label = self.objects[gt_subject_label]
			boxes.append(gt_subject_bbox)
			labels.append(self._class_to_ind[gt_subject_label])

			gt_object_bbox = y1y2x1x2_to_x1y1x2y2(gt_object_bbox)
			gt_object_label = self.objects[gt_object_label]
			boxes.append(gt_object_bbox)
			labels.append(self._class_to_ind[gt_object_label])
		return boxes, labels

	def __getitem__(self, index):
		img_name = self.imgs_list[index]
		boxes, labels = self.load_pascal_annotation(img_name)
		img_path = self.image_path_from_index(img_name)
		img = Image.open(img_path)
		img = torch.from_numpy(np.array(img))
		img = img.permute(2,0,1)
		img = img.type(torch.float32)

		assert len(boxes) == len(labels), "boxes and labels should be of equal length"

		return {'boxes': torch.tensor(boxes, dtype=torch.float32),
				'labels': torch.tensor(labels, dtype=torch.int64),
				'img': img
				# 'preds': preds,
				}
				

def collater(data):
	imgs = [s['img'] for s in data]
	annotations = [{"boxes": s['boxes'].to(DEVICE)} for s in data]
	for i, s in enumerate(data):
		annotations[i]['labels'] = s['labels'].to(DEVICE)
	return imgs, annotations
