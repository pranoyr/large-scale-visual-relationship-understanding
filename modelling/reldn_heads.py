import numpy as np
from numpy import linalg as la
import math
import logging
import json

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from .word_vector import get_obj_prd_vecs
# import nn as mynn


# import utils.net as net_utils
# from modeling.sparse_targets_rel import FrequencyBias

logger = logging.getLogger(__name__)


class reldn_head(nn.Module):
	def __init__(self, dim_in):
		super().__init__()
		# initialize word vectors
		self.obj_vecs, self.prd_vecs = get_obj_prd_vecs()

		# add subnet
		self.prd_feats = nn.Sequential(
			nn.Linear(dim_in, 1024),
			nn.LeakyReLU(0.1))
		self.prd_vis_embeddings = nn.Sequential(
			nn.Linear(1024 * 3, 1024),
			nn.LeakyReLU(0.1),
			nn.Linear(1024, 1024))
		self.prd_sem_hidden = nn.Sequential(
			nn.Linear(300, 1024),
			nn.LeakyReLU(0.1),
			nn.Linear(1024, 1024))
		self.prd_sem_embeddings = nn.Sequential(
				nn.Linear(300, 1024),
				nn.LeakyReLU(0.1),
				nn.Linear(1024, 1024))
		
		self.so_vis_embeddings = nn.Linear(dim_in // 3, 1024)
		self.so_sem_embeddings = nn.Sequential(
			nn.Linear(300, 1024),
			nn.LeakyReLU(0.1),
			nn.Linear(1024, 1024))
			
	def forward(self, spo_feat=None, sbj_feat=None, obj_feat=None):
		device = sbj_feat.device
		sbj_vis_embeddings = self.so_vis_embeddings(sbj_feat)
		obj_vis_embeddings = self.so_vis_embeddings(obj_feat)
		
		prd_hidden = self.prd_feats(spo_feat)
		prd_features = torch.cat((sbj_vis_embeddings.detach(), prd_hidden, obj_vis_embeddings.detach()), dim=1)
		prd_vis_embeddings = self.prd_vis_embeddings(prd_features)

		ds_obj_vecs = self.obj_vecs
		ds_obj_vecs = Variable(torch.from_numpy(ds_obj_vecs.astype('float32'))).to(device)
		so_sem_embeddings = self.so_sem_embeddings(ds_obj_vecs)
		so_sem_embeddings = F.normalize(so_sem_embeddings, p=2, dim=1)  # (#prd, 1024)
		so_sem_embeddings.t_()

		sbj_vis_embeddings = F.normalize(sbj_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
		sbj_sim_matrix = torch.mm(sbj_vis_embeddings, so_sem_embeddings)  # (#bs, #prd)
		sbj_cls_scores = 3 * sbj_sim_matrix
		
		obj_vis_embeddings = F.normalize(obj_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
		obj_sim_matrix = torch.mm(obj_vis_embeddings, so_sem_embeddings)  # (#bs, #prd)
		obj_cls_scores = 3 * obj_sim_matrix

		ds_prd_vecs = self.prd_vecs
		ds_prd_vecs = Variable(torch.from_numpy(ds_prd_vecs.astype('float32'))).to(device)
		prd_sem_embeddings = self.prd_sem_embeddings(ds_prd_vecs)
		prd_sem_embeddings = F.normalize(prd_sem_embeddings, p=2, dim=1)  # (#prd, 1024)
		prd_vis_embeddings = F.normalize(prd_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
		prd_sim_matrix = torch.mm(prd_vis_embeddings, prd_sem_embeddings.t_())  # (#bs, #prd)
		prd_cls_scores = 3 * prd_sim_matrix
		

		if not self.training:
			sbj_cls_scores = F.softmax(sbj_cls_scores, dim=1)
			obj_cls_scores = F.softmax(obj_cls_scores, dim=1)
			prd_cls_scores = F.softmax(prd_cls_scores, dim=1)

		return sbj_cls_scores, obj_cls_scores, prd_cls_scores
