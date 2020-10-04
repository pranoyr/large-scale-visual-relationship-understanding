# from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models.detection._utils as  det_utils
from torchvision.ops import boxes as box_ops
import os
from losses import fastrcnn_loss, reldn_losses
from collections import OrderedDict
import random
import torch
from . import reldn_heads
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
import time
import pdb
from torchvision.models.resnet import resnet101
import torchvision
import math
from torch.jit.annotations import Optional, List, Dict, Tuple
from torchvision.models.detection.faster_rcnn import MultiScaleRoIAlign, TwoMLPHead, FastRCNNPredictor 
from config import cfg


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

		representation_size = 1024
		self.box_predictor = FastRCNNPredictor(
			representation_size,
			cfg.NUM_CLASSES)

		self.RelDN = reldn_heads.reldn_head(self.box_head.fc7.out_features * 3)  # concat of SPO

		self.box_similarity = box_ops.box_iou
		# assign ground-truth boxes for each proposal
		self.proposal_matcher = det_utils.Matcher(
			cfg.FG_IOU_THRESH,
			cfg.BG_IOU_THRESH,
			allow_low_quality_matches=False)

		self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
			cfg.BATCH_SIZE_PER_IMAGE,
			cfg.POSITIVE_FRACTION)

		self.fg_bg_sampler_so = det_utils.BalancedPositiveNegativeSampler(
			cfg.BATCH_SIZE_PER_IMAGE_SO,
			cfg.POSITIVE_FRACTION_SO)
			
		bbox_reg_weights = (10., 10., 5., 5.)
		self.box_coder = det_utils.BoxCoder(bbox_reg_weights)


	def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, assign_to='all'):
		# type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
		matched_idxs = []
		labels = []
		if assign_to == "subject":
			slice_index = 0
		elif assign_to =='all':
			slice_index = -1
		else:
			slice_index = 1
		for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
			
			if slice_index >= 0:
				gt_boxes_in_image = gt_boxes_in_image[:,slice_index,:]
				gt_labels_in_image = gt_labels_in_image[:,slice_index]
			else:
				gt_boxes_in_image = gt_boxes_in_image.view(-1,4)
				gt_labels_in_image = gt_labels_in_image.view(-1)
				
			if gt_boxes_in_image.numel() == 0:
				# Background image
				device = proposals_in_image.device
				clamped_matched_idxs_in_image = torch.zeros(
					(proposals_in_image.shape[0],), dtype=torch.int64, device=device
				)
				labels_in_image = torch.zeros(
					(proposals_in_image.shape[0],), dtype=torch.int64, device=device
				)
			else:
				#  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
				match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
				matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

				clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

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
		else:
			sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler_so(labels)
		sampled_inds = []
		for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
			zip(sampled_pos_inds, sampled_neg_inds)
		):
			img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
			sampled_inds.append(img_sampled_inds)
		return sampled_inds

	def add_gt_proposals(self, proposals, gt_boxes):
		# type: (List[Tensor], List[Tensor]) -> List[Tensor]
		proposals = [
			torch.cat((proposal, gt_box.view(-1,4)))
			for proposal, gt_box in zip(proposals, gt_boxes)
		]

		return proposals

	def extract_positive_proposals(self, data):
		n_props = []
		n_labels = []
		proposals = data['proposals']
		labels = data['labels']
		for proposal, label in zip(proposals, labels):
			mask = label > 0
			pos_label = label[mask]
			pos_proposal = proposal[mask]

			n_labels.append(pos_label)
			n_props.append(pos_proposal)
			
		return {'labels':n_labels, 'proposals':n_props}

	def check_targets(self, targets):
		# type: (Optional[List[Dict[str, Tensor]]]) -> None
		assert targets is not None
		assert all(["boxes" in t for t in targets])
		assert all(["labels" in t for t in targets])

	def select_training_samples(self,
								proposals,  # type: List[Tensor]
								targets     # type: Optional[List[Dict[str, Tensor]]]
								):
		# type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
		self.check_targets(targets)
		assert targets is not None
		dtype = proposals[0].dtype
		device = proposals[0].device

		gt_boxes = [t["boxes"].to(dtype) for t in targets]  # shape  --> list of [Tensor of size 10,2,4]
		gt_labels = [t["labels"] for t in targets]   		# shape  --> list of [Tensor of size 10,2]
		# gt_preds = [t["preds"] for t in targets]

		# append ground-truth bboxes to propos
		proposals = self.add_gt_proposals(proposals, gt_boxes)


		# get matching gt indices for each proposal
		matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, assign_to="all")
		# sample a fixed proportion of positive-negative proposals
		sampled_inds = self.subsample(labels, sample_for="all")								# size 512
		matched_gt_boxes = []
		num_images = len(proposals)
		for img_id in range(num_images):
			img_sampled_inds = sampled_inds[img_id]
			proposals[img_id] = proposals[img_id][img_sampled_inds]
			labels[img_id] = labels[img_id][img_sampled_inds]
			matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

			gt_boxes_in_image = gt_boxes[img_id].view(-1,4)
			if gt_boxes_in_image.numel() == 0:
				gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
			matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

		regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
		data = {"labels":labels, "proposals":proposals}
		data = self.extract_positive_proposals(data)
		pos_proposals = data['proposals']
		
	
		# get matching gt indices for each proposal
		sub_matched_idxs, sub_labels = self.assign_targets_to_proposals(pos_proposals, gt_boxes, gt_labels, assign_to="subject")
		sampled_inds = self.subsample(sub_labels, sample_for="subject")   			#	size 64 --> 32 pos, 32 neg
		sub_proposals = pos_proposals.copy()
		sub_matched_gt_boxes = []
		num_images = len(proposals)
		for img_id in range(num_images):
			img_sampled_inds = sampled_inds[img_id]
			sub_proposals[img_id] = pos_proposals[img_id][img_sampled_inds]
			sub_labels[img_id] = sub_labels[img_id][img_sampled_inds]
			sub_matched_idxs[img_id] = sub_matched_idxs[img_id][img_sampled_inds]

			gt_boxes_in_image = gt_boxes[img_id][:,0,:]
			if gt_boxes_in_image.numel() == 0:
				gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
			sub_matched_gt_boxes.append(gt_boxes_in_image[sub_matched_idxs[img_id]])

		sub_regression_targets = self.box_coder.encode(sub_matched_gt_boxes, sub_proposals)
		data_s = {"labels":sub_labels, "proposals":sub_proposals}
		data_s = self.extract_positive_proposals(data_s)

		

		# get matching gt indices for each proposal
		obj_matched_idxs, obj_labels = self.assign_targets_to_proposals(pos_proposals, gt_boxes, gt_labels, assign_to="objects")
		sampled_inds = self.subsample(obj_labels, sample_for="object")   				#size 64 --> 32 pos, 32 neg
		obj_proposals = pos_proposals.copy()
		obj_matched_gt_boxes = []
		num_images = len(proposals)
		for img_id in range(num_images):
			img_sampled_inds = sampled_inds[img_id]
			obj_proposals[img_id] = pos_proposals[img_id][img_sampled_inds]
			obj_labels[img_id] = obj_labels[img_id][img_sampled_inds]
			obj_matched_idxs[img_id] = obj_matched_idxs[img_id][img_sampled_inds]

			gt_boxes_in_image = gt_boxes[img_id][:,1,:]
			if gt_boxes_in_image.numel() == 0:
				gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
			obj_matched_gt_boxes.append(gt_boxes_in_image[obj_matched_idxs[img_id]])

		obj_regression_targets = self.box_coder.encode(obj_matched_gt_boxes, obj_proposals)
		data_o = {"labels":obj_labels, "proposals":obj_proposals}
		data_o = self.extract_positive_proposals(data_o)


		# prepare relation candidates
		# relation_proposals = create_relation_proposals(sub_proposals, obj_proposals)   size --> 64 * 64 = 4096 relation proposals
		# assign predicate to subjects
		# matched_idxs, sub_labels = self.assign_targets_to_relation_proposals(relation_proposals, gt_boxes, gt_labels)
		# assign predicate to objects
		# matched_idxs, obj_labels = self.assign_targets_to_proposals(obj_proposals, obj_regression_targets, predicates)
		# compare sub_labels == obj_labels 
		# gt_predicates =    of size   List[tensor[] of size 512 * 512 ]
		# return gt_predicates, sub_obj_proposals

		return proposals, matched_idxs, labels, regression_targets, data_s, data_o

	def postprocess_detections(self,
							   class_logits,    # type: Tensor
							   box_regression,  # type: Tensor
							   proposals,       # type: List[Tensor]
							   image_shapes     # type: List[Tuple[int, int]]
							   ):
		# type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
		device = class_logits.device
		num_classes = class_logits.shape[-1]

		boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
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
			inds = torch.nonzero(scores > cfg.SCORE_THRESH).squeeze(1)
			boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

			# remove empty boxes
			keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
			boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

			# non-maximum suppression, independently done per class
			keep = box_ops.batched_nms(boxes, scores, labels, cfg.NMS_THRESH)
			# keep only topk scoring predictions
			keep = keep[:cfg.DETECTIONS_PER_IMG]
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
			proposals, matched_idxs, labels, regression_targets, data_s, data_o = self.select_training_samples(proposals, targets)
			# print(labels[0].shape)
		else:
			labels = None
			regression_targets = None
			matched_idxs = None

		# faster_rcnn branch
		box_features = self.box_roi_pool(features, proposals, image_shapes)
		box_features = self.box_head(box_features)
		class_logits, box_regression = self.box_predictor(box_features)
		
		# predicate branch
		sbj_feat = self.box_roi_pool(features, data_s["proposals"], image_shapes)
		sbj_feat = self.box_head(sbj_feat)
		obj_feat = self.box_roi_pool(features, data_o["proposals"], image_shapes)
		obj_feat = self.box_head(obj_feat)

		#rel_feat = self.Prd_RCNN.Box_Head(blob_conv_prd, rel_ret, rois_name='rel_rois', use_relu=use_relu)
		#concat_feat = torch.cat((sbj_feat, rel_feat, obj_feat), dim=1)
		#prd_cls_scores, sbj_cls_scores, obj_cls_scores = \
			#    self.RelDN(concat_feat, sbj_labels, obj_labels, sbj_feat, obj_feat)

		sbj_cls_scores, obj_cls_scores = \
				self.RelDN(sbj_feat, obj_feat)

		result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
		losses = {}
		if self.training:
			assert labels is not None and regression_targets is not None

			loss_cls_sbj, accuracy_cls_sbj = reldn_losses(sbj_cls_scores, data_s["labels"])
			loss_cls_obj, accuracy_cls_obj = reldn_losses(obj_cls_scores, data_o['labels'])

			loss_classifier, loss_box_reg = fastrcnn_loss(
				class_logits, box_regression, labels, regression_targets)
			losses = {
				"loss_classifier": loss_classifier,
				"loss_box_reg": loss_box_reg,
				"loss_sbj" : loss_cls_sbj,
				"acc_sbj"	: accuracy_cls_sbj.item(),
				"loss_obj" : loss_cls_obj,
				"acc_obj"	: accuracy_cls_obj.item()
			}
		else:
			boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
			num_images = len(boxes)
			for i in range(num_images):
				result.append(
					{
						"boxes": boxes[i],
						"labels": labels[i],
						"scores": scores[i],
					}
				)
		   
		return result, losses


			