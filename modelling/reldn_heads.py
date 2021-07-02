import json
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from config import cfg
from numpy import linalg as la
from torch import nn
from torch.autograd import Variable
from torch.nn import init

from .word_vector import get_obj_prd_vecs

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
        if not cfg.MODEL.USE_SEM_CONCAT:
            self.prd_sem_embeddings = nn.Sequential(
                nn.Linear(300, 1024),
                nn.LeakyReLU(0.1),
                nn.Linear(1024, 1024))
        else:
            self.prd_sem_hidden = nn.Sequential(
                nn.Linear(300, 1024),
                nn.LeakyReLU(0.1),
                nn.Linear(1024, 1024))
            self.prd_sem_embeddings = nn.Linear(3 * 1024, 1024)

        self.so_vis_embeddings = nn.Linear(dim_in // 3, 1024)
        self.so_sem_embeddings = nn.Sequential(
            nn.Linear(300, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, spo_feat=None, sbj_feat=None, obj_feat=None, targets=None, sbj_labels=None, obj_labels=None):
        device = sbj_feat.device
        sbj_vis_embeddings = self.so_vis_embeddings(sbj_feat)
        obj_vis_embeddings = self.so_vis_embeddings(obj_feat)

        prd_hidden = self.prd_feats(spo_feat)
        prd_features = torch.cat(
            (sbj_vis_embeddings.detach(), prd_hidden, obj_vis_embeddings.detach()), dim=1)
        prd_vis_embeddings = self.prd_vis_embeddings(prd_features)

        ds_obj_vecs = self.obj_vecs
        ds_obj_vecs = Variable(torch.from_numpy(
            ds_obj_vecs.astype('float32'))).to(device)
        so_sem_embeddings = self.so_sem_embeddings(ds_obj_vecs)
        so_sem_embeddings = F.normalize(
            so_sem_embeddings, p=2, dim=1)  # (#prd, 1024)
        so_sem_embeddings.t_()

        sbj_vis_embeddings = F.normalize(
            sbj_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
        sbj_sim_matrix = torch.mm(
            sbj_vis_embeddings, so_sem_embeddings)  # (#bs, #prd)
        sbj_cls_scores = cfg.MODEL.NORM_SCALE * sbj_sim_matrix

        obj_vis_embeddings = F.normalize(
            obj_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
        obj_sim_matrix = torch.mm(
            obj_vis_embeddings, so_sem_embeddings)  # (#bs, #prd)
        obj_cls_scores = cfg.MODEL.NORM_SCALE * obj_sim_matrix

        if not cfg.MODEL.USE_SEM_CONCAT:
            ds_prd_vecs = self.prd_vecs
            ds_prd_vecs = Variable(torch.from_numpy(
                ds_prd_vecs.astype('float32'))).to(device)
            prd_sem_embeddings = self.prd_sem_embeddings(ds_prd_vecs)
            prd_sem_embeddings = F.normalize(
                prd_sem_embeddings, p=2, dim=1)  # (#prd, 1024)
            prd_vis_embeddings = F.normalize(
                prd_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
            prd_sim_matrix = torch.mm(
                prd_vis_embeddings, prd_sem_embeddings.t_())  # (#bs, #prd)
            prd_cls_scores = cfg.MODEL.NORM_SCALE * prd_sim_matrix
        else:
            ds_prd_vecs = self.prd_vecs
            ds_prd_vecs = Variable(torch.from_numpy(ds_prd_vecs.astype('float32'))).to(device)
            prd_sem_hidden = self.prd_sem_hidden(ds_prd_vecs)  # (#prd, 1024)
            # get sbj vis embeddings and expand to (#bs, #prd, 1024)
            sbj_vecs = self.obj_vecs[sbj_labels.cpu().numpy()]  # (#bs, 300)
            sbj_vecs = Variable(torch.from_numpy(sbj_vecs.astype('float32'))).to(device)
            if len(list(sbj_vecs.size())) == 1:  # sbj_vecs should be 2d
                sbj_vecs.unsqueeze_(0)
            sbj_sem_embeddings = self.so_sem_embeddings(sbj_vecs)  # (#bs, 1024)
            sbj_sem_embeddings = sbj_sem_embeddings.unsqueeze(1).expand(
                sbj_vecs.shape[0], ds_prd_vecs.shape[0], 1024)  # (#bs, 1024) --> # (#bs, 1, 1024) --> # (#bs, #prd, 1024)
            # get obj vis embeddings and expand to (#bs, #prd, 1024)
            obj_vecs = self.obj_vecs[obj_labels.cpu().numpy()]  # (#bs, 300)
            obj_vecs = Variable(torch.from_numpy(obj_vecs.astype('float32'))).to(device)
            if len(list(obj_vecs.size())) == 1:  # obj_vecs should be 2d
                obj_vecs.unsqueeze_(0)
            obj_sem_embeddings = self.so_sem_embeddings(obj_vecs)  # (#bs, 1024)
            obj_sem_embeddings = obj_sem_embeddings.unsqueeze(1).expand(
                obj_vecs.shape[0], ds_prd_vecs.shape[0], 1024)  # (#bs, 1024) --> # (#bs, 1, 1024) --> # (#bs, #prd, 1024)
            # expand prd hidden feats to (#bs, #prd, 1024)
            prd_sem_hidden = prd_sem_hidden.unsqueeze(0).expand(
                sbj_vecs.shape[0], ds_prd_vecs.shape[0], 1024)  # (#prd, 1024) --> # (1, #prd, 1024) --> # (#bs, #prd, 1024)
            # now feed semantic SPO features into the last prd semantic layer
            spo_sem_feat = torch.cat(
                (sbj_sem_embeddings.detach(),
                 prd_sem_hidden,
                 obj_sem_embeddings.detach()),
                dim=2)  # (#bs, #prd, 3 * 1024)
            # get prd scores
            prd_sem_embeddings = self.prd_sem_embeddings(spo_sem_feat)  # (#bs, #prd, 1024)
            prd_sem_embeddings = F.normalize(prd_sem_embeddings, p=2, dim=2)  # (#bs, #prd, 1024)
            prd_vis_embeddings = F.normalize(prd_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
            prd_vis_embeddings = prd_vis_embeddings.unsqueeze(-1)  # (#bs, 1024) --> (#bs, 1024, 1)
            prd_sim_matrix = torch.bmm(prd_sem_embeddings, prd_vis_embeddings).squeeze(-1)  # bmm((#bs, #prd, 1024), (#bs, 1024, 1)) = (#bs, #prd, 1) --> (#bs, #prd)
            prd_cls_scores = cfg.MODEL.NORM_SCALE * prd_sim_matrix

        if not targets:
            sbj_cls_scores = F.softmax(sbj_cls_scores, dim=1)
            obj_cls_scores = F.softmax(obj_cls_scores, dim=1)
            prd_cls_scores = F.softmax(prd_cls_scores, dim=1)

        return sbj_cls_scores, obj_cls_scores, prd_cls_scores
