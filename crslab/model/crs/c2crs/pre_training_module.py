# @Time    :   2022/1/1
# @Author  :   Yuanhang Zhou
# @email   :   sdzyh002@gmail.com

import ipdb
import math
import torch
import numpy as np
import torch.nn.functional as F
from loguru import logger
from torch import nn

from collections import defaultdict
import os
from crslab.model.base import BaseModel
from crslab.model.utils.modules.info_nce_loss import info_nce_loss
from crslab.model.utils.functions import edge_to_pyg_format
from crslab.model.utils.modules.cross_entropy_loss import Handle_Croess_Entropy_Loss

from torch_geometric.nn import RGCNConv
from crslab.config import MODEL_PATH

from crslab.model.base import BaseModel
from crslab.model.utils.modules.info_nce_loss import info_nce_loss
from crslab.model.utils.functions import edge_to_pyg_format
from crslab.model.utils.modules.attention import SelfAttentionBatch, SelfAttentionSeq
from crslab.model.utils.modules.transformer import TransformerDecoder, TransformerEncoder
from crslab.model.utils.modules.info_nce_loss import info_nce_loss, masked_info_nce_loss


class CoarseToFinePretrainModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self._build_model()
        
    def _build_model(self):
        self.cross_entory_loss = nn.CrossEntropyLoss()
    
    def CoarseFineLoss(
            self, 
            kg_user_rep, context_user_rep, review_user_rep, 
            ekg_reps, eic_reps, er_reps,
            ekg_e_ids, eic_e_ids, er_e_ids,
            ekg_conv_ids, eic_conv_ids, er_conv_ids
            ):

        coarseLoss = self.CoarseLoss(
            kg_user_rep, context_user_rep, review_user_rep, 
            ekg_reps, eic_reps, er_reps,
            ekg_e_ids, eic_e_ids, er_e_ids,
            ekg_conv_ids, eic_conv_ids, er_conv_ids)

        fineLoss = self.FineLoss(
            kg_user_rep, context_user_rep, review_user_rep, 
            ekg_reps, eic_reps, er_reps,
            ekg_e_ids, eic_e_ids, er_e_ids,
            ekg_conv_ids, eic_conv_ids, er_conv_ids)

        loss = (coarseLoss + fineLoss) / 2
        return loss
    
    def WeightCoarseFineLoss(
            self, 
            kg_user_rep, context_user_rep, review_user_rep, 
            ekg_reps, eic_reps, er_reps,
            ekg_e_ids, eic_e_ids, er_e_ids,
            ekg_conv_ids, eic_conv_ids, er_conv_ids
            ):

        coarseLoss = self.CoarseLoss(
            kg_user_rep, context_user_rep, review_user_rep, 
            ekg_reps, eic_reps, er_reps,
            ekg_e_ids, eic_e_ids, er_e_ids,
            ekg_conv_ids, eic_conv_ids, er_conv_ids)

        fineLoss = self.FineLoss(
            kg_user_rep, context_user_rep, review_user_rep, 
            ekg_reps, eic_reps, er_reps,
            ekg_e_ids, eic_e_ids, er_e_ids,
            ekg_conv_ids, eic_conv_ids, er_conv_ids)

        loss = (self.config.coarse_loss_lambda*coarseLoss + self.config.fine_loss_lambda*fineLoss) / 2.0
        return loss

    def CoarseLoss(
            self, 
            kg_user_rep, context_user_rep, review_user_rep, 
            ekg_reps, eic_reps, er_reps,
            ekg_e_ids, eic_e_ids, er_e_ids,
            ekg_conv_ids, eic_conv_ids, er_conv_ids
            ):
        assert kg_user_rep.shape == context_user_rep.shape
        assert kg_user_rep.shape == review_user_rep.shape

        # user level cl
        kg_c_loss = self._get_info_nce_loss(kg_user_rep, context_user_rep)
        kg_r_loss = self._get_info_nce_loss(kg_user_rep, review_user_rep)
        c_r_loss = self._get_info_nce_loss(context_user_rep, review_user_rep)

        loss = (kg_c_loss + kg_r_loss + c_r_loss) / 3.0

        return loss
    
    def FineLoss(
            self, 
            kg_user_rep, context_user_rep, review_user_rep, 
            ekg_reps, eic_reps, er_reps,
            ekg_e_ids, eic_e_ids, er_e_ids,
            ekg_conv_ids, eic_conv_ids, er_conv_ids
            ):
        assert ekg_reps.shape[0] == len(ekg_e_ids)
        assert eic_reps.shape[0] == len(eic_e_ids)
        assert er_reps.shape[0] == len(er_e_ids)
        assert len(ekg_conv_ids) == len(ekg_e_ids)
        assert len(eic_conv_ids) == len(eic_e_ids)
        assert len(er_conv_ids) == len(er_e_ids)

        # entity level cl
        ekg_eic_sample_ids, ekg_eic_co_occur_mask, view1_reps, view2_reps = self._del_single_rep(
            ekg_e_ids, eic_e_ids, ekg_conv_ids, eic_conv_ids, ekg_reps, eic_reps)
        ekg_eic_loss = self._get_masked_info_nce_loss(
            view1_reps, 
            view2_reps, 
            ekg_eic_sample_ids,
            ekg_eic_co_occur_mask
            )

        ekg_er_sample_ids, ekg_er_co_occur_mask, view1_reps, view2_reps = self._del_single_rep(
            ekg_e_ids, er_e_ids, ekg_conv_ids, er_conv_ids, ekg_reps, er_reps)
        ekg_er_loss = self._get_masked_info_nce_loss(
            view1_reps,
            view2_reps,
            ekg_er_sample_ids, 
            ekg_er_co_occur_mask
            )

        eic_er_sample_ids, eic_er_co_occur_mask, view1_reps, view2_reps = self._del_single_rep(
            eic_e_ids, er_e_ids, eic_e_ids, er_conv_ids, eic_reps, er_reps)
        eic_er_loss = self._get_masked_info_nce_loss(
            view1_reps,
            view2_reps,
            eic_er_sample_ids, 
            eic_er_co_occur_mask
            )

        loss = (ekg_eic_loss + ekg_er_loss + eic_er_loss) / 3.0

        return loss

    def _del_single_rep(self, view1_e_ids, view2_e_ids, view1_conv_ids, view2_conv_ids, view1_reps, view2_reps):
        view1_e_ids = [int(eid) for eid in view1_e_ids]
        view2_e_ids = [int(eid) for eid in view2_e_ids]

        sample_ids = view1_e_ids + view2_e_ids
        co_occur_mask = view1_conv_ids + view2_conv_ids

        eid2count = defaultdict(int)
        for eid in sample_ids:
            eid2count[eid] += 1

        save_index, view1_save_index, view2_save_index = [], [], []
        for i, eid in enumerate(sample_ids):
            if eid2count[eid] > 1:
                save_index.append(i)
        for i, eid in enumerate(view1_e_ids):
            if eid2count[eid] > 1:
                view1_save_index.append(i)
        for i, eid in enumerate(view2_e_ids):
            if eid2count[eid] > 1:
                view2_save_index.append(i)

        sample_ids = torch.LongTensor(sample_ids).to(self.config.device)
        co_occur_mask = torch.LongTensor(co_occur_mask).to(self.config.device)

        sample_ids = sample_ids[save_index]
        co_occur_mask = co_occur_mask[save_index]
        view1_reps = view1_reps[view1_save_index]
        view2_reps = view2_reps[view2_save_index]

        return sample_ids, co_occur_mask, view1_reps, view2_reps

    def _get_masked_info_nce_loss(self, user_rep1, user_rep2, sample_ids, co_occur_mask):
        # user_rep1 = (bs, dim), user_rep2 = (bs, dim)
        features = torch.cat([user_rep1, user_rep2], dim=0) # (2*bs, dim)
        batch_size = user_rep1.shape[0]

        logits, labels = masked_info_nce_loss(
            features, 
            sample_ids, 
            co_occur_mask,
            self.config.device, 
            self.config.temperature
            )
        
        kg_r_loss = self.cross_entory_loss(logits, labels)

        return kg_r_loss
    
    def forward(self, user_model, batch, mode, epoch):
        loss = self.coarse_to_fine_pretrain(user_model, batch, mode, epoch)
        return loss

    def weight_pretrain(self, user_model, batch, mode, epoch):
        # contrastive learning for pretraining
        entity_ids_in_context = batch['entity_ids_in_context']
        entities_has_distInfo = batch['entities_has_distInfo']
        eic_conv_ids = batch['eic_conv_ids']
        er_conv_ids = batch['er_conv_ids']

        kg_user_rep, context_user_rep, review_user_rep, \
            entity_kg_reps, entity_context_text_reps, entities_review_rep, \
                duplicate_removal_ekg_ids, duplicate_removal_ekg_conv_ids \
                    = user_model.get_project_user_rep(batch, mode)

        loss = self.WeightCoarseFineLoss(
            kg_user_rep, context_user_rep, review_user_rep, 
            entity_kg_reps, entity_context_text_reps, entities_review_rep,
            duplicate_removal_ekg_ids, entity_ids_in_context, entities_has_distInfo,
            duplicate_removal_ekg_conv_ids, eic_conv_ids, er_conv_ids)

        return loss
    
    def coarse_to_fine_pretrain(self, user_model, batch, mode, epoch):
        # contrastive learning for pretraining
        entity_ids_in_context = batch['entity_ids_in_context']
        entities_has_distInfo = batch['entities_has_distInfo']
        eic_conv_ids = batch['eic_conv_ids']
        er_conv_ids = batch['er_conv_ids']
        # ipdb.set_trace()
        kg_user_rep, context_user_rep, review_user_rep, \
            entity_kg_reps, entity_context_text_reps, entities_review_rep, \
                duplicate_removal_ekg_ids, duplicate_removal_ekg_conv_ids \
                    = user_model.get_project_user_rep(batch, mode)

        # ipdb.set_trace()
        if epoch <= self.config.coarse_pretrain_epoch:
            loss = self.CoarseLoss(
                kg_user_rep, context_user_rep, review_user_rep, 
                entity_kg_reps, entity_context_text_reps, entities_review_rep,
                duplicate_removal_ekg_ids, entity_ids_in_context, entities_has_distInfo,
                duplicate_removal_ekg_conv_ids, eic_conv_ids, er_conv_ids)
        else:
            loss = self.WeightCoarseFineLoss(
                kg_user_rep, context_user_rep, review_user_rep, 
                entity_kg_reps, entity_context_text_reps, entities_review_rep,
                duplicate_removal_ekg_ids, entity_ids_in_context, entities_has_distInfo,
                duplicate_removal_ekg_conv_ids, eic_conv_ids, er_conv_ids)

        return loss

    def _get_info_nce_loss(self, user_rep1, user_rep2):
        # user_rep1 = (bs, dim), user_rep2 = (bs, dim)
        features = torch.cat([user_rep1, user_rep2], dim=0) # (2*bs, dim)
        batch_size = user_rep1.shape[0]

        logits, labels = info_nce_loss(
            features, 
            bs=batch_size, 
            n_views=2, 
            device=self.config.device, 
            temperature=self.config.temperature)
        
        kg_r_loss = self.cross_entory_loss(logits, labels)

        return kg_r_loss
    