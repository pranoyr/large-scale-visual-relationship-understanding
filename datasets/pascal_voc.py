from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import pathlib
import xml.etree.ElementTree as ET
import skimage.io
import skimage.transform
import skimage.color
import skimage
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


def _findNode(parent, name, debug_name=None, parse=None):
	if debug_name is None:
		debug_name = name

	result = parent.find(name)
	if result is None:
		raise ValueError('missing element \'{}\''.format(debug_name))
	if parse is not None:
		try:
			return parse(result.text)
		except ValueError as e:
			raise_from(ValueError(
				'illegal value for \'{}\': {}'.format(debug_name, e)), None)
	return result


class VOCDataset:
	"""VOC Dataset"""

	def __init__(self, root, transform=None, is_test=False, keep_difficult=False):
		""" Args:
				root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
																		Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
		"""
		self.root = pathlib.Path(root)
		self.transform = transform
		if is_test:
			image_sets_file = self.root / "ImageSets/Main/test.txt"
		else:
			image_sets_file = self.root / "ImageSets/Main/trainval.txt"
		self.ids = VOCDataset._read_image_ids(image_sets_file)
		self.keep_difficult = keep_difficult
		# if the labels file exists, read in the class names
		label_file_name = self.root / "labels.txt"

		if os.path.isfile(label_file_name):
			class_string = ""
			with open(label_file_name, 'r') as infile:
				for line in infile:
					class_string += line.rstrip()

			# classes should be a comma separated list
			classes = class_string.split(',')
			# prepend BACKGROUND as first class
			classes.insert(0, 'BACKGROUND')
			classes = [elem.replace(" ", "") for elem in classes]
			self.class_names = tuple(classes)
			print("VOC Labels read from file: " + str(self.class_names))

		else:
			print("No labels file, using default VOC classes.")
			self.class_names = ('__background__','aeroplane', 'bicycle', 'bird', 'boat',
								'bottle', 'bus', 'car', 'cat', 'chair',
								'cow', 'diningtable', 'dog', 'horse',
								'motorbike', 'person', 'pottedplant',
								'sheep', 'sofa', 'train', 'tvmonitor')

		self.class_dict = {class_name: i for i,
						   class_name in enumerate(self.class_names)}

		# remove ids
		self.remove_ids()

	def num_classes(self):
		return len(self.class_names)

	def remove_ids(self):
		mask = []
		image_names_copy = self.ids.copy()
		for image_name in self.ids:
			filename = os.path.join(image_name + '.xml')
			tree = ET.parse(os.path.join(self.root, 'Annotations', filename))
			annotations = self.__parse_annotations(tree.getroot())
			mask.append(annotations['labels'].size != 0)

		mask = np.array(mask)
		self.ids = np.array(self.ids)
		self.ids = self.ids[mask]

	def __parse_annotation(self, element):
		""" Parse an annotation given an XML element.
		"""
		# truncated = _findNode(element, 'truncated', parse=int)
		# difficult = _findNode(element, 'difficult', parse=int)

		class_name = _findNode(element, 'name').text
		if class_name not in self.class_dict:
			return None, None

		box = []
		label = self.class_dict[class_name]

		bndbox = _findNode(element, 'bndbox')
		box.append(_findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1)
		box.append(_findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1)
		box.append(_findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1)
		box.append(_findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1)

		return box, label

	def __parse_annotations(self, xml_root):
		""" Parse all annotations under the xml_root.
		"""
		annotations = {'labels': [], 'bboxes': []}

		for i, element in enumerate(xml_root.iter('object')):
			try:
				box, label = self.__parse_annotation(
					element)
				if label is None:
					continue
			except ValueError as e:
				raise_from(ValueError(
					'could not parse object #{}: {}'.format(i, e)), None)

			annotations['bboxes'].append(np.array(box))
			annotations['labels'].append(label)

		annotations['bboxes'] = np.array(annotations['bboxes'])
		annotations['labels'] = np.array(annotations['labels'])

		return annotations

	def __getitem__(self, index):
		image_id = self.ids[index]
		boxes, labels, _ = self._get_annotation(image_id)
		# if not self.keep_difficult:
		# 	boxes = boxes[is_difficult == 0]
		# 	labels = labels[is_difficult == 0]
		img = self.load_image(image_id)
		sample = {'img': img, 'annot': boxes, 'labels':labels}
		# if self.transform:
		# 	sample = self.transform(sample)

		return sample

	def get_annotation(self, index):
		image_id = self.ids[index]
		return image_id, self._get_annotation(image_id)

	def __len__(self):
		return len(self.ids)

	@staticmethod
	def _read_image_ids(image_sets_file):
		ids = []
		with open(image_sets_file) as f:
			for line in f:
				ids.append(line.rstrip())
		return ids

	def _get_annotation(self, image_id):
		annotation_file = self.root / f"Annotations/{image_id}.xml"
		objects = ET.parse(annotation_file).findall("object")
		boxes = []
		labels = []
		is_difficult = []
		for object in objects:
			class_name = object.find('name').text.lower().strip()
			# we're only concerned with clases in our list
			if class_name in self.class_dict:
				bbox = object.find('bndbox')

				# VOC dataset format follows Matlab, in which indexes start from 0
				x1 = float(bbox.find('xmin').text) - 1
				y1 = float(bbox.find('ymin').text) - 1
				x2 = float(bbox.find('xmax').text) - 1
				y2 = float(bbox.find('ymax').text) - 1
				boxes.append([x1, y1, x2, y2])

				labels.append(self.class_dict[class_name])
				is_difficult_str = object.find('difficult').text
				is_difficult.append(int(is_difficult_str)
									if is_difficult_str else 0)

		return (torch.tensor(boxes, dtype=torch.float32),
				torch.tensor(labels, dtype=torch.float32),
				torch.tensor(is_difficult, dtype=torch.uint8))

	def load_image(self, image_id):
		image_file = self.root / f"JPEGImages/{image_id}.jpg"
		img = Image.open(image_file)
		img = torch.from_numpy(np.array(img))
		img = img.permute(2,0,1)
		img = img.type(torch.float32)
		return img

	def image_aspect_ratio(self, index):
		image_id = self.ids[index]
		image_file = self.root / f"JPEGImages/{image_id}.jpg"
		image = Image.open(image_file)
		return float(image.width) / float(image.height)


# def collater(data):
# 	imgs = [s['img'] for s in data]
# 	annotations = [{"boxes": s['annot'].cuda()} for s in data]
# 	return imgs, annotations



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def collater(data):
	imgs = [s['img'] for s in data]
	annotations = [{"boxes": s['annot'].to(DEVICE)} for s in data]
	for i, s in enumerate(data):
		annotations[i]['labels'] = s['labels'].type(torch.int64).to(DEVICE)
	# labels = [{"labels": s['labels']} for s in data]
	return imgs, annotations


# def collater(data):
# 	imgs = [s['img'] for s in data]
# 	annotations = [{"boxes": s['annot']} for s in data]
# 	for i, s in enumerate(data):
# 		annotations[i]['labels'] = s['labels'].type(torch.int64)
# 	# labels = [{"labels": s['labels']} for s in data]
# 	return imgs, annotations

