import copy
import math
import os
import pdb
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.models.detection._utils as det_utils
import torchvision.utils as vutils
import utils.boxes as box_utils
from config import cfg
from losses import fastrcnn_loss, reldn_losses
from torch.jit.annotations import Dict, List, Optional, Tuple
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import (FastRCNNPredictor,
													  GeneralizedRCNNTransform,
													  MultiScaleRoIAlign,
													  TwoMLPHead)
from torchvision.models.resnet import resnet101
from torchvision.ops import boxes as box_ops

from . import reldn_heads


class RoIHeads(torch.nn.Module):
	__annotations__ = {
		'box_coder': det_utils.BoxCoder,
		'proposal_matcher': det_utils.Matcher,
		'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
	}

	def __init__(self):
		super(RoIHeads, self).__init__()

		self.box_roi_pool = MultiScaleRoIAlign(
			featmap_names=['0', '1', '2', '3'],
			output_size=7,
			sampling_ratio=2)

		resolution = self.box_roi_pool.output_size[0]
		representation_size = 1024
		self.box_head = TwoMLPHead(
			256 * resolution ** 2,
			representation_size)
		self.rlp_head = copy.deepcopy(self.box_head)

		representation_size = 1024
		self.box_predictor = FastRCNNPredictor(
			representation_size,
			cfg.BOX.NUM_CLASSES)

		self.RelDN = reldn_heads.reldn_head(
			self.box_head.fc7.out_features * 3)  # concat of SPO

		self.box_similarity = box_ops.box_iou
		# assign ground-truth boxes for each proposal
		self.proposal_matcher = det_utils.Matcher(
			cfg.BOX.FG_IOU_THRESH,
			cfg.BOX.BG_IOU_THRESH,
			allow_low_quality_matches=False)

		self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
			cfg.BOX.BATCH_SIZE_PER_IMAGE,
			cfg.BOX.POSITIVE_FRACTION)

		self.fg_bg_sampler_so = det_utils.BalancedPositiveNegativeSampler(
			cfg.MODEL.BATCH_SIZE_PER_IMAGE_SO,
			cfg.MODEL.POSITIVE_FRACTION_SO)

		self.fg_bg_sampler_rlp = det_utils.BalancedPositiveNegativeSampler(
			cfg.MODEL.BATCH_SIZE_PER_IMAGE_REL,
			cfg.MODEL.POSITIVE_FRACTION_REL)

		bbox_reg_weights = (10., 10., 5., 5.)
		self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

	def assign_pred_to_rlp_proposals(self, sbj_proposals, obj_proposals, gt_boxes, gt_labels, gt_preds):
		# type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
		labels = []
		for sbj_proposals_in_image, obj_proposals_in_image, gt_boxes_in_image, gt_labels_in_image, \
				gt_preds_in_image in zip(sbj_proposals, obj_proposals, gt_boxes, gt_labels, gt_preds):

			# Remove dulplicates for sbj and obj gths
			gt_sbj_boxes_in_image = torch.unique(gt_boxes_in_image[:, 0, :], dim=0)
			gt_obj_boxes_in_image = torch.unique(gt_boxes_in_image[:, 1, :], dim=0)
			
			sbj_match_quality_matrix = box_ops.box_iou(
				gt_sbj_boxes_in_image, sbj_proposals_in_image)
			obj_match_quality_matrix = box_ops.box_iou(
				gt_obj_boxes_in_image, obj_proposals_in_image)
			sbj_matched_idxs_in_image = self.proposal_matcher(
				sbj_match_quality_matrix)
			obj_matched_idxs_in_image = self.proposal_matcher(
				obj_match_quality_matrix)

			sbj_boxes = gt_sbj_boxes_in_image[sbj_matched_idxs_in_image]
			obj_boxes = gt_obj_boxes_in_image[obj_matched_idxs_in_image]
			labels_in_image = torch.zeros(sbj_boxes.shape[0])
			for i in range(len(sbj_boxes)):
				sbj_indices = torch.where(torch.all(gt_boxes_in_image[:, 0, :] == sbj_boxes[i], dim=1))[0]
				obj_indices = torch.where(torch.all(gt_boxes_in_image[:, 1, :] == obj_boxes[i], dim=1))[0]
				matched_idx = np.intersect1d(sbj_indices.cpu().numpy(), obj_indices.cpu().numpy())
				if matched_idx.any():
					labels_in_image[i] = gt_preds_in_image[matched_idx[0]]

			labels_in_image = labels_in_image.to(dtype=torch.int64, device=torch.device(cfg.DEVICE))
			labels.append(labels_in_image)
		return labels

	def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, assign_to='all'):
		# type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
		matched_idxs = []
		labels = []
		if assign_to == "subject":
			slice_index = 0
		elif assign_to == 'all':
			slice_index = -1
		else:
			slice_index = 1
		for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

			if slice_index >= 0:
				gt_boxes_in_image = gt_boxes_in_image[:, slice_index, :]
				gt_labels_in_image = gt_labels_in_image[:, slice_index]
			else:
				gt_boxes_in_image = gt_boxes_in_image.view(-1, 4)
				gt_labels_in_image = gt_labels_in_image.view(-1)

			if gt_boxes_in_image.numel() == 0:
				# Background image
				device = proposals_in_image.device
				clamped_matched_idxs_in_image = torch.zeros(
					(proposals_in_image.shape[0],
					 ), dtype=torch.int64, device=device
				)
				labels_in_image = torch.zeros(
					(proposals_in_image.shape[0],
					 ), dtype=torch.int64, device=device
				)
			else:
				#  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
				match_quality_matrix = box_ops.box_iou(
					gt_boxes_in_image, proposals_in_image)
				matched_idxs_in_image = self.proposal_matcher(
					match_quality_matrix)

				clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(
					min=0)

				labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
				labels_in_image = labels_in_image.to(dtype=torch.int64)

				# Label background (below the low threshold)
				bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
				labels_in_image[bg_inds] = 0

				# Label ignore proposals (between low and high thresholds)
				ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
				labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

			matched_idxs.append(clamped_matched_idxs_in_image)
			labels.append(labels_in_image)
		return matched_idxs, labels

	def subsample(self, labels, sample_for="all"):
		# type: (List[Tensor]) -> List[Tensor]
		if sample_for == "all":
			sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
		elif sample_for == "rel":
			sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler_rlp(labels)
		else:
			sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler_so(labels)
		sampled_inds = []
		for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
			zip(sampled_pos_inds, sampled_neg_inds)
		):
			img_sampled_inds = torch.nonzero(
				pos_inds_img | neg_inds_img).squeeze(1)
			sampled_inds.append(img_sampled_inds)
		return sampled_inds

	def add_gt_proposals(self, proposals, gt_boxes):
		# type: (List[Tensor], List[Tensor]) -> List[Tensor]
		proposals = [
			torch.cat((proposal, gt_box.view(-1, 4)))
			for proposal, gt_box in zip(proposals, gt_boxes)
		]

		return proposals

	def remove_self_pairs(self, sbj_inds, obj_inds):
		mask = sbj_inds != obj_inds
		return sbj_inds[mask], obj_inds[mask]

	def extract_positive_proposals(self, labels, proposals):
		n_props = []
		n_labels = []
		for proposal, label in zip(proposals, labels):
			mask = label > 0
			pos_label = label[mask]
			pos_proposal = proposal[mask]
			n_labels.append(pos_label)
			n_props.append(pos_proposal)

		return n_labels, n_props

	def combine_labels(self, pos_sbj_labels, pos_obj_labels):
		all_labels = []
		for sbj_labels, obj_labels in zip(pos_sbj_labels, pos_obj_labels):
			all_labels.append(torch.cat([sbj_labels, obj_labels]))
		return all_labels

	def combine_boxes(self, pos_sbj_boxes, pos_obj_boxes):
		all_boxes = []
		for sbj_boxes, obj_boxes in zip(pos_sbj_boxes, pos_obj_boxes):
    		print("***")
			print(sbj_boxes.shape)
			print(obj_boxes.shape)
			print(torch.cat([sbj_boxes, obj_boxes]).shape)
			all_boxes.append(torch.cat([sbj_boxes, obj_boxes]))
		return all_boxes

	def check_targets(self, targets):
		# type: (Optional[List[Dict[str, Tensor]]]) -> None
		assert targets is not None
		assert all(["boxes" in t for t in targets])
		assert all(["labels" in t for t in targets])

	def select_training_samples(self,
								proposals,  # type: List[Tensor]
								# type: Optional[List[Dict[str, Tensor]]]
								targets
								):
		# type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
		self.check_targets(targets)
		assert targets is not None
		dtype = proposals[0].dtype
		device = proposals[0].device

		# shape  --> list of [Tensor of size 10,2,4]
		gt_boxes = [t["boxes"].to(dtype) for t in targets]
		# shape  --> list of [Tensor of size 10,2]
		gt_labels = [t["labels"] for t in targets]
		gt_preds = [t["preds"] for t in targets]

		# append ground-truth bboxes to propos
		proposals = self.add_gt_proposals(proposals, gt_boxes)

		# get matching gt indices for each proposal
		matched_idxs, labels = self.assign_targets_to_proposals(
			proposals, gt_boxes, gt_labels, assign_to="all")
		# sample a fixed proportion of positive-negative proposals
		sampled_inds = self.subsample(labels, sample_for="all")  # size 512
		all_proposals = proposals.copy()
		matched_gt_boxes = []
		num_images = len(proposals)
		for img_id in range(num_images):
			img_sampled_inds = sampled_inds[img_id]
			all_proposals[img_id] = proposals[img_id][img_sampled_inds]
			labels[img_id] = labels[img_id][img_sampled_inds]
			matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

			gt_boxes_in_image = gt_boxes[img_id].view(-1, 4)
			if gt_boxes_in_image.numel() == 0:
				gt_boxes_in_image = torch.zeros(
					(1, 4), dtype=dtype, device=device)
			matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

		regression_targets = self.box_coder.encode(
			matched_gt_boxes, all_proposals)

		# get matching gt indices for each proposal
		_, sbj_labels = self.assign_targets_to_proposals(
			proposals, gt_boxes, gt_labels, assign_to="subject")
		sampled_inds = self.subsample(
			sbj_labels, sample_for="subject")   			  # 64 --> 32 pos, 32 neg
		sbj_proposals = proposals.copy()
		for img_id in range(num_images):
			img_sampled_inds = sampled_inds[img_id]
			sbj_proposals[img_id] = proposals[img_id][img_sampled_inds]
			sbj_labels[img_id] = sbj_labels[img_id][img_sampled_inds]
		pos_sbj_labels, pos_sbj_proposals = self.extract_positive_proposals(
			sbj_labels, sbj_proposals)

		# get matching gt indices for each proposal
		_, obj_labels = self.assign_targets_to_proposals(
			proposals, gt_boxes, gt_labels, assign_to="objects")
		sampled_inds = self.subsample(
			obj_labels, sample_for="object")   				# 64 --> 32 pos, 32 neg
		obj_proposals = proposals.copy()
		for img_id in range(num_images):
			img_sampled_inds = sampled_inds[img_id]
			obj_proposals[img_id] = proposals[img_id][img_sampled_inds]
			obj_labels[img_id] = obj_labels[img_id][img_sampled_inds]
		pos_obj_labels, pos_obj_proposals = self.extract_positive_proposals(
			obj_labels, obj_proposals)

		all_labels = self.combine_labels(pos_sbj_labels, pos_obj_labels)
		all_proposals = self.combine_boxes(pos_sbj_proposals, pos_obj_proposals)
		
		# prepare relation proposals
		rlp_proposals = []
		for img_id in range(num_images):
			sbj_shape = all_labels[img_id].shape[0]
			obj_shape = all_labels[img_id].shape[0]
			sbj_inds = np.repeat(np.arange(sbj_shape), obj_shape)
			obj_inds = np.tile(np.arange(obj_shape), sbj_shape)

			pos_sbj_labels[img_id] = all_labels[img_id][sbj_inds]
			pos_obj_labels[img_id] = all_labels[img_id][obj_inds]
			pos_sbj_proposals[img_id] = all_proposals[img_id][sbj_inds]
			pos_obj_proposals[img_id] = all_proposals[img_id][obj_inds]

			rlp_proposals.append(box_utils.boxes_union(
				pos_obj_proposals[img_id], pos_sbj_proposals[img_id]))

		# assign gt_predicate to relation proposals
		rlp_labels = self.assign_pred_to_rlp_proposals(pos_sbj_proposals, pos_obj_proposals,
													   gt_boxes, gt_labels, gt_preds)

		# 128 --> 64 pos, 64 neg)
		sampled_inds = self.subsample(rlp_labels, sample_for="rel")
		for img_id in range(num_images):
			img_sampled_inds = sampled_inds[img_id]
			pos_sbj_proposals[img_id] = pos_sbj_proposals[img_id][img_sampled_inds]
			pos_obj_proposals[img_id] = pos_obj_proposals[img_id][img_sampled_inds]
			rlp_proposals[img_id] = rlp_proposals[img_id][img_sampled_inds]
			pos_sbj_labels[img_id] = pos_sbj_labels[img_id][img_sampled_inds]-1
			pos_obj_labels[img_id] = pos_obj_labels[img_id][img_sampled_inds]-1
			rlp_labels[img_id] = rlp_labels[img_id][img_sampled_inds]

		data_sbj = {'proposals': pos_sbj_proposals, 'labels': pos_sbj_labels}
		data_obj = {'proposals': pos_obj_proposals, 'labels': pos_obj_labels}
		data_rlp = {'proposals': rlp_proposals, 'labels': rlp_labels}

		print(pos_sbj_proposals[0].shape)
		print(pos_obj_proposals[0].shape)
		print(rlp_proposals[0].shape)


		return all_proposals, matched_idxs, labels, regression_targets, data_sbj, data_obj, data_rlp

	def postprocess_detections(self,
							   class_logits,    # type: Tensor
							   box_regression,  # type: Tensor
							   proposals,       # type: List[Tensor]
							   image_shapes     # type: List[Tuple[int, int]]
							   ):
		# type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
		device = class_logits.device
		num_classes = class_logits.shape[-1]

		boxes_per_image = [boxes_in_image.shape[0]
						   for boxes_in_image in proposals]
		pred_boxes = self.box_coder.decode(box_regression, proposals)

		pred_scores = F.softmax(class_logits, -1)

		pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
		pred_scores_list = pred_scores.split(boxes_per_image, 0)

		all_boxes = []
		all_scores = []
		all_labels = []
		for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
			boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

			# create labels for each prediction
			labels = torch.arange(num_classes, device=device)
			labels = labels.view(1, -1).expand_as(scores)

			# remove predictions with the background label
			boxes = boxes[:, 1:]
			scores = scores[:, 1:]
			labels = labels[:, 1:]

			# batch everything, by making every class prediction be a separate instance
			boxes = boxes.reshape(-1, 4)
			scores = scores.reshape(-1)
			labels = labels.reshape(-1)

			# remove low scoring boxes
			inds = torch.nonzero(scores > cfg.BOX.SCORE_THRESH).squeeze(1)
			boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

			# remove empty boxes
			keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
			boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

			# non-maximum suppression, independently done per class
			keep = box_ops.batched_nms(boxes, scores, labels, cfg.BOX.NMS_THRESH)
			# keep only topk scoring predictions
			keep = keep[:cfg.BOX.DETECTIONS_PER_IMG]
			boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

			all_boxes.append(boxes)
			all_scores.append(scores)
			all_labels.append(labels)

		return all_boxes, all_scores, all_labels

	def forward(self,
				features,      # type: Dict[str, Tensor]
				proposals,     # type: List[Tensor]
				image_shapes,  # type: List[Tuple[int, int]]
				targets=None   # type: Optional[List[Dict[str, Tensor]]]
				):
		# type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
		"""
		Arguments:
						features (List[Tensor])
						proposals (List[Tensor[N, 4]])
						image_shapes (List[Tuple[H, W]])
						targets (List[Dict])
		"""
		if targets is not None:
			for t in targets:
				# TODO: https://github.com/pytorch/pytorch/issues/26731
				floating_point_types = (torch.float, torch.double, torch.half)
				assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
				assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
		if self.training:
			proposals, matched_idxs, labels, regression_targets, data_sbj, data_obj, data_rlp = self.select_training_samples(
				proposals, targets)

			# faster_rcnn branch
			box_features = self.box_roi_pool(features, proposals, image_shapes)
			box_features = self.box_head(box_features)
			class_logits, box_regression = self.box_predictor(box_features)

			# predicate branch
			sbj_feat = self.box_roi_pool(
				features, data_sbj["proposals"], image_shapes)
			sbj_feat = self.box_head(sbj_feat)
			obj_feat = self.box_roi_pool(
				features, data_obj["proposals"], image_shapes)
			obj_feat = self.box_head(obj_feat)

			rel_feat = self.box_roi_pool(
				features, data_rlp["proposals"], image_shapes)
			rel_feat = self.rlp_head(rel_feat)

			concat_feat = torch.cat((sbj_feat, rel_feat, obj_feat), dim=1)

			sbj_cls_scores, obj_cls_scores, rlp_cls_scores = \
				self.RelDN(concat_feat, sbj_feat, obj_feat)

			result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
			losses = {}

			assert labels is not None and regression_targets is not None

			loss_cls_sbj, accuracy_cls_sbj = reldn_losses(
				sbj_cls_scores, data_sbj["labels"])
			loss_cls_obj, accuracy_cls_obj = reldn_losses(
				obj_cls_scores, data_obj['labels'])
			loss_cls_rlp, accuracy_cls_rlp = reldn_losses(
				rlp_cls_scores, data_rlp['labels'])

			loss_classifier, loss_box_reg = fastrcnn_loss(
				class_logits, box_regression, labels, regression_targets)
			losses = {
				"loss_classifier": loss_classifier,
				"loss_box_reg": loss_box_reg,
				"loss_sbj": loss_cls_sbj,
				"acc_sbj"	: accuracy_cls_sbj.item(),
				"loss_obj": loss_cls_obj,
				"acc_obj"	: accuracy_cls_obj.item(),
				"loss_rlp": loss_cls_rlp,
				"acc_rlp"	: accuracy_cls_rlp.item()
			}

		else:
			labels = None
			regression_targets = None
			matched_idxs = None
			result = []

			# faster_rcnn branch
			box_features = self.box_roi_pool(features, proposals, image_shapes)
			box_features = self.box_head(box_features)
			class_logits, box_regression = self.box_predictor(box_features)

			boxes, scores, labels = self.postprocess_detections(
				class_logits, box_regression, proposals, image_shapes)
			num_images = len(boxes)

			all_sbj_boxes = []
			all_obj_boxes = []
			all_rlp_boxes = []
			all_shapes = []
			for img_id in range(num_images):
				sbj_inds = np.repeat(
					np.arange(boxes[img_id].shape[0]), boxes[img_id].shape[0])
				obj_inds = np.tile(
					np.arange(boxes[img_id].shape[0]), boxes[img_id].shape[0])

				sbj_inds, obj_inds = self.remove_self_pairs(sbj_inds, obj_inds)

				sbj_boxes = boxes[img_id][sbj_inds]
				obj_boxes = boxes[img_id][obj_inds]
				rlp_boxes = box_utils.boxes_union(sbj_boxes, obj_boxes)

				all_sbj_boxes.append(sbj_boxes)
				all_obj_boxes.append(obj_boxes)
				all_rlp_boxes.append(rlp_boxes)
				all_shapes.append(rlp_boxes.shape[0])

			# predicate branch
			sbj_feat = self.box_roi_pool(features, all_sbj_boxes, image_shapes)
			sbj_feat = self.box_head(sbj_feat)
			obj_feat = self.box_roi_pool(features, all_obj_boxes, image_shapes)
			obj_feat = self.box_head(obj_feat)
			rel_feat = self.box_roi_pool(features, all_rlp_boxes, image_shapes)
			rel_feat = self.rlp_head(rel_feat)
			concat_feat = torch.cat((sbj_feat, rel_feat, obj_feat), dim=1)
			sbj_cls_scores, obj_cls_scores, rlp_cls_scores = \
				self.RelDN(concat_feat, sbj_feat, obj_feat)

			sbj_cls_scores_list, obj_cls_scores_list, rlp_cls_scores_list = \
				sbj_cls_scores.split(all_shapes), obj_cls_scores.split(
					all_shapes), rlp_cls_scores.split(all_shapes)

			for i, _ in enumerate(sbj_cls_scores_list):
				_, sbj_indices = torch.max(sbj_cls_scores_list[i], dim=1)
				_, obj_indices = torch.max(obj_cls_scores_list[i], dim=1)
				rel_scores, rel_indices = torch.max(rlp_cls_scores_list[i], dim=1)
				# filter "unknown"
				mask = rel_indices > 0
				rel_scores = rel_scores[mask]
				predicates = rel_indices[mask]
				subjects = sbj_indices[mask]
				objects = obj_indices[mask]

				sbj_boxes = all_sbj_boxes[i][mask]
				obj_boxes = all_obj_boxes[i][mask]
				rlp_boxes = all_rlp_boxes[i][mask]

				score_mask = rel_scores > cfg.TEST.THRESHOLD
				result = [{"sbj_boxes": sbj_boxes[score_mask],
						   "obj_boxes": obj_boxes[score_mask],
						   'sbj_labels': subjects[score_mask],
						   'obj_labels': objects[score_mask],
						   'predicates': predicates[score_mask],
						   }]
				# result = [{"sbj_boxes": sbj_boxes,
				#            "obj_boxes": obj_boxes,
				#            'sbj_labels': subjects,
				#            'obj_labels': objects,
				#            'predicates': predicates,
				#            }]
				losses = {}

		return result, losses
