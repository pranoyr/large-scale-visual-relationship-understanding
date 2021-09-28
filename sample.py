# import json
# from utils.boxes import xywh_to_xyxy
# from config import cfg
# import gensim


# def get_obj_prd_vecs():
#     word_vector_path = cfg.WORD_VECTORS_DIR
#     dataset_path = cfg.DATASET_DIR
#     word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
#         word_vector_path, binary=True)
#     print('Word Embeddings loaded.')
#     # change everything into lowercase
#     all_keys = list(word2vec_model.vocab.keys())
#     for key in all_keys:
#         new_key = key.lower()
#         word2vec_model.vocab[new_key] = word2vec_model.vocab.pop(key)
#     print('Wiki words converted to lowercase.')
#     return word2vec_model


# word2vec_model = get_obj_prd_vecs()


# import numpy.linalg as la
# def check_word_vector(obj_cat):
#     a = 0
#     obj_words = obj_cat.split()
#     for word in obj_words:
#         try:
#             raw_vec = word2vec_model[word]
#             x = raw_vec / la.norm(raw_vec)
#             a = a + x
#         except:
#             return False
#     y = a/len(obj_words)
#     return True

# # Opening JSON file
# f = open('./data/VG/json_dataset/relationships.json')

# # returns JSON object as
# # a dictionary
# data = json.load(f)
# print(len(data))

# objects = []
# predicates = []

# def load_annotations(index):
#     for spo in data[index]['relationships']:
#         try:
#             gt_sbj_label = spo['subject']['name']
#         except:
#             gt_sbj_label = ''.join(spo['subject']['names'][0])

#         # gt_sbj_bbox = spo['subject']['bbox']
#         try:
#             gt_obj_label = spo['object']['name']
#         except:
#             gt_obj_label = ''.join(spo['object']['names'][0])
#         # gt_obj_bbox = spo['object']['bbox']
#         predicate = spo['predicate']

#         if gt_sbj_label and check_word_vector(gt_sbj_label):
#             objects.append(gt_sbj_label)
#         if gt_obj_label and check_word_vector(gt_obj_label):
#             objects.append(gt_obj_label)
#         if predicate and check_word_vector(predicate):
#             predicates.append(predicate)
#         # return(gt_sbj_label , predicate, gt_obj_label)

#         # prepare bboxes for subject and object
#         # gt_sbj_bbox = y1y2x1x2_to_x1y1x2y2(gt_sbj_bbox)
#         # gt_obj_bbox = y1y2x1x2_to_x1y1x2y2(gt_obj_bbox)
#         # boxes.append([gt_sbj_bbox, gt_obj_bbox])

#         # prepare labels for subject and object
#         # map to word
#         # gt_sbj_label = self.all_objects[gt_sbj_label]
#         # gt_obj_label = self.all_objects[gt_obj_label]
#         # predicate = self.predicates[predicate]
#         # # map to new index
#         # labels.append([self._class_to_ind[gt_sbj_label],
#         #                 self._class_to_ind[gt_obj_label]])
#         # preds.append(self._preds_to_ind[predicate])


# def __getitem__(index):
#     img_name = data[index]['image_id']
#     load_annotations(index)
#     # img_path = self.image_path_from_index(img_name)
#     # img = Image.open(img_path)
#     # img = self.transform(img)


# for i in range(len(data)):
#     __getitem__(i)

# objects = list(set(objects))
# predicates = list(set(predicates))
# print(len(objects))
# print(len(predicates))
# with open('objects.json', 'w') as f:
#     json.dump(objects, f)

# with open('predicates.json', 'w') as f:
#     json.dump(predicates, f)



import math
def batch_images(images, size_divisible=3):
	# concatenate
	max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))

	stride = size_divisible
	max_size = list(max_size)
	max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
	max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
	max_size = tuple(max_size)

	batch_shape = (len(images),) + max_size
	batched_imgs = images[0].new(*batch_shape).zero_()
	for img, pad_img in zip(images, batched_imgs):
		pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

	return batched_imgs

import torch
import numpy as np
import cv2
from PIL import Image
img1 = Image.open('/Users/pranoyr/Desktop/vrd_sample2/6438875_deae648618_b.jpg')
img2 = Image.open('/Users/pranoyr/Desktop/vrd_sample2/6438875_deae648618_b.jpg')
img2 = img2.resize((500, 500))
img1 = torch.Tensor(np.array(img1))
img2 = torch.Tensor(np.array(img2))
images = [img1, img2]
images = batch_images(images)
print(images.shape)
cv2.imshow('img', images[1].numpy())
cv2.waitKey(0)