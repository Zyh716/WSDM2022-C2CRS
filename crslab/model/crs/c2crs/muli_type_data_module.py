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

   
class CoarseFineKGModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self._build_model()
        
    def _build_model(self):
        self._build_kg_layer()
        self._build_kg_cl_project_head()
        self._build_ekg_cl_project_head()

    def _build_kg_layer(self):
        # print([self.config.n_entity, self.config.kg_emb_dim, self.config.n_relation, self.config.num_bases])
        # ipdb.set_trace()
        self.kg_encoder = RGCNConv(self.config.n_entity, self.config.kg_emb_dim, self.config.n_relation, num_bases=self.config.num_bases)
        self.kg_attn = SelfAttentionBatch(self.config.kg_emb_dim, self.config.kg_emb_dim)
        self.kg_user_rep_dense = nn.Linear(self.config.kg_emb_dim, self.config.user_emb_dim)

    def _build_kg_cl_project_head(self):
        self.kg_project_head_fc1 = nn.Linear(self.config.kg_emb_dim, self.config.kg_emb_dim)
        self.kg_project_head_fc2 = nn.Linear(self.config.kg_emb_dim, self.config.user_emb_dim)

    def _build_ekg_cl_project_head(self):
        # entity-level kg
        self.ekg_project_head_fc1 = nn.Linear(self.config.kg_emb_dim, self.config.kg_emb_dim)
        self.ekg_project_head_fc2 = nn.Linear(self.config.kg_emb_dim, self.config.user_emb_dim)

    def get_project_kg_rep(self, batch, mode):
        context_entities = batch['context_entities']
        entity_ids_in_context = batch['entity_ids_in_context'] # [bs*n_eic]
        eic_conv_ids = batch['eic_conv_ids'] # [bs*n_eic]
        
        duplicate_removal_ekg_ids, duplicate_removal_ekg_conv_ids = self.get_duplicate_removal_ids(entity_ids_in_context, eic_conv_ids) # [<=bs*n_eic], [<=bs*n_eic]

        kg_user_rep, kg_embedding = self._get_kg_user_rep(context_entities)  # (bs, dim), (n_entity, dim)
        project_kg_user_rep = self._project_kg_user_rep(kg_user_rep)  # (bs, dim)

        project_entity_kg_reps = self._get_project_entity_kg_reps(kg_embedding, duplicate_removal_ekg_ids) # (~bs*n_eic, user_dim) or (1, user_dim)

        return project_kg_user_rep, duplicate_removal_ekg_ids, duplicate_removal_ekg_conv_ids, project_entity_kg_reps # (bs, dim), (~bs*n_eic, user_dim) or (1, user_dim)
    
    def get_duplicate_removal_ids(self, entity_ids_in_context, eic_conv_ids):
        # [bs*n_eic], [bs*n_eic]
        dr_ekg_ids_set, duplicate_removal_ekg_ids, duplicate_removal_ekg_conv_ids = set(), [], []

        for ekg_id, ekg_conv_id in zip(entity_ids_in_context, eic_conv_ids):
            if ekg_id not in dr_ekg_ids_set:
                dr_ekg_ids_set.add(ekg_id)
                duplicate_removal_ekg_ids.append(ekg_id)
                duplicate_removal_ekg_conv_ids.append(ekg_conv_id)

        return duplicate_removal_ekg_ids, duplicate_removal_ekg_conv_ids # [<=bs*n_eic], [<=bs*n_eic]

    def _get_project_entity_kg_reps(self, kg_embedding, entity_ids):
        entity_kg_reps = self._get_entity_kg_reps(kg_embedding, entity_ids) # (~b*n_eic, kg_dim) or (1, kg_dim)
        project_entity_kg_reps = self._project_entity_kg_reps(entity_kg_reps)  # (~b*n_eic, user_dim) or (1, user_dim)

        return project_entity_kg_reps  # (~b*n_eic, user_dim) or (1, user_dim)

    def _get_entity_kg_reps(self, kg_embedding, entity_ids):
        # (n_entity, kg_dim),

        if len(entity_ids) == 0:
            entity_kg_reps = torch.zeros((1, self.kg_emb_dim))
        else:
            entity_kg_reps = kg_embedding[entity_ids] # (~bs*n_eic, kg_dim)

        return entity_kg_reps # (~bs*n_eic, kg_dim) or (1, kg_dim)
    
    def _project_entity_kg_reps(self, entity_kg_reps):
        entity_kg_reps = self.ekg_project_head_fc1(entity_kg_reps)
        entity_kg_reps = F.relu(entity_kg_reps)
        entity_kg_reps = self.ekg_project_head_fc2(entity_kg_reps)

        return entity_kg_reps

    def get_kg_rep(self, batch, mode):
        context_entities = batch['context_entities']

        kg_user_rep, kg_embedding = self._get_kg_user_rep(context_entities)  # (bs, dim), (n_entity, dim)

        return kg_user_rep, kg_embedding

    def _get_kg_user_rep(self, context_entities):
        # ipdb.set_trace()
        kg_embedding = self.kg_encoder(None, self.config.edge_idx, self.config.edge_type)
        # print(self.config.edge_idx.shape)
        # ipdb.set_trace()
        user_rep = self._encode_user(context_entities, kg_embedding)  # (bs, dim)

        return user_rep, kg_embedding

    def _encode_user(self, entity_lists, kg_embedding):
        user_repr_list = []
        for entity_list in entity_lists:
            if not entity_list:
                user_repr_list.append(torch.zeros(self.config.user_emb_dim, device=self.config.device))
                continue
            user_repr = kg_embedding[entity_list]
            user_repr = self.kg_attn(user_repr)
            user_repr_list.append(user_repr)
        return torch.stack(user_repr_list, dim=0)  # (bs, dim)

    def _project_kg_user_rep(self, kg_user_rep):
        kg_user_rep = self.kg_project_head_fc1(kg_user_rep) # (bs, dim)
        kg_user_rep = F.relu(kg_user_rep) # (bs, dim)
        kg_user_rep = self.kg_project_head_fc2(kg_user_rep) # (bs, dim)

        return kg_user_rep # (bs, dim)
   

class CoarseFineContextModel(nn.Module):
    def __init__(self, config, token_embedding):
        super().__init__()
        self.config = config
        self.token_embedding = token_embedding

        self._build_model()

    def _build_model(self):
        self._build_context_transformer_layer()
        self._build_context_cl_project_head()
        self._build_atten()
        self._build_ect_cl_project_head()
        
    def _build_context_transformer_layer(self):
        self.register_buffer('C_START', torch.tensor([self.config.start_token_idx], dtype=torch.long))

        self.context_encoder = TransformerEncoder(
            self.config.n_heads,
            self.config.n_layers,
            self.config.token_emb_dim,
            self.config.ffn_size,
            self.config.vocab_size,
            self.token_embedding,
            self.config.dropout,
            self.config.attention_dropout,
            self.config.relu_dropout,
            self.config.pad_token_idx,
            self.config.learn_positional_embeddings,
            self.config.embeddings_scale,
            self.config.reduction,
            self.config.n_positions
        )

        self.context_user_rep_dense = nn.Linear(self.config.token_emb_dim, self.config.user_emb_dim)

    def _build_context_cl_project_head(self):
        self.context_project_head_fc1 = nn.Linear(self.config.token_emb_dim, self.config.token_emb_dim)
        self.context_project_head_fc2 = nn.Linear(self.config.token_emb_dim, self.config.user_emb_dim)

    def _build_atten(self):
        self.ContextHiddenStateAttenFunc = SelfAttentionSeq(self.config.token_emb_dim, self.config.token_emb_dim)

    def _build_ect_cl_project_head(self):
        # entity-level context text
        self.ect_project_head_fc1 = nn.Linear(self.config.token_emb_dim, self.config.token_emb_dim)
        self.ect_project_head_fc2 = nn.Linear(self.config.token_emb_dim, self.config.user_emb_dim)

    def get_project_context_rep(self, batch, mode):
        context = batch['context']  # (bs, seq_len)
        context_mask = batch['context_mask']  # (bs, seq_len)
        context_pad_mask = batch['context_pad_mask']  # (bs, seq_len)
        entity_masks_in_context = batch['entity_masks_in_context'] # [bs]*(n_eic, seq_len), n_eic maybe equal to 0
        entity_ids_in_context = batch['entity_ids_in_context'] # [~bs*n_eic]

        context_user_rep, context_state = self._get_context_user_rep(context, context_mask, context_pad_mask) # (bs, dim), (bs, seq_len, dim)
        project_context_user_rep = self._project_context_user_rep(context_user_rep)  # (bs, dim)

        project_entity_context_text_reps = self._get_project_entity_context_text_reps(context_state, entity_masks_in_context) 
        # (~bs*n_eic, user_dim) or (1, user_dim)
        assert len(entity_ids_in_context) == project_entity_context_text_reps.shape[0]

        return project_context_user_rep, project_entity_context_text_reps # (bs, dim), (~bs*n_eic, user_dim) or (1, user_dim)
    
    def _get_project_entity_context_text_reps(self, context_state, entity_masks_in_context):
        # (bs, seq_len, dim), [bs]*(n_eic, seq_len)
        entity_context_text_reps = self._get_entity_context_text_reps(context_state, entity_masks_in_context) # (~bs*n_eic, tok_dim) or (1, tok_dim)
        project_entity_context_text_reps = self._project_entity_context_text_reps(entity_context_text_reps) # (~b*n_eic, user_dim) or (1, user_dim)

        return project_entity_context_text_reps # (~b*n_eic, user_dim) or (1, user_dim)

    def _get_entity_context_text_reps(self, context_state, entity_masks_in_context):
        #  (bs, seq_len, tok_dim), [bs]*(n_eic, seq_len), n_eic maybe equal to 0
        entity_context_text_reps = []

        for sample_context_state, sample_entity_masks_in_context in zip(context_state, entity_masks_in_context):
            # (seq_len, tok_dim), (n_eic, seq_len) or ()
            if sample_entity_masks_in_context.shape[0] == 0:
                continue
            sample_context_state = sample_context_state.unsqueeze(0) # (1, seq_len, tok_dim)
            sample_entity_masks_in_context = sample_entity_masks_in_context.to(self.config.device)
            sample_entity_masks_in_context = sample_entity_masks_in_context.unsqueeze(2) # (n_eic, seq_len, 1)
            masked_sample_context_state = sample_context_state * sample_entity_masks_in_context # (n_eic, seq_len, tok_dim)
            sample_entity_context_text_reps = torch.sum(masked_sample_context_state, dim=1) # (n_eic, tok_dim)
            entity_context_text_reps.append(sample_entity_context_text_reps)
        
        if len(entity_context_text_reps) == 0:
            entity_context_text_reps = torch.zeros((1, self.config.token_emb_dim))
        else:
            entity_context_text_reps = torch.cat(entity_context_text_reps, dim=0) # (~bs*n_eic, tok_dim)

        return entity_context_text_reps # (~bs*n_eic, tok_dim) or (1, tok_dim)
    
    def _project_entity_context_text_reps(self, entity_context_text_reps):
        entity_context_text_reps = self.ect_project_head_fc1(entity_context_text_reps)
        entity_context_text_reps = F.relu(entity_context_text_reps)
        entity_context_text_reps = self.ect_project_head_fc2(entity_context_text_reps)

        return entity_context_text_reps
    
    def _get_context_user_rep(self, context, context_mask, context_pad_mask):
        cls_state, state = self._get_hidden_state_context_transformer(context)  # (bs, dim), (bs, seq_len, dim)
        atten_last_state = self.ContextHiddenStateAttenFunc(state, context_pad_mask)  # (bs, dim)

        assert len(atten_last_state.shape) == 2
        return atten_last_state, state  # (bs, dim), (bs, seq_len, dim)

    def _get_hidden_state_context_transformer(self, context):
        state, mask = self.context_encoder(context)
        cls_state = state[:, 0, :] # (bs, dim)

        return cls_state, state

    def _project_context_user_rep(self, context_user_rep):
        # context_user_rep = (bs, dim)
        context_user_rep = self.context_project_head_fc1(context_user_rep) # (bs, dim)
        context_user_rep = F.relu(context_user_rep) # (bs, dim)
        context_user_rep = self.context_project_head_fc2(context_user_rep) # (bs, dim)

        return context_user_rep # (bs, dim)


class CoarseFineDistReviewModel(nn.Module):
    def __init__(self, config, token_embedding):
        super().__init__()
        self.config = config
        self.token_embedding = token_embedding

        self._build_model()
    
    def _build_model(self):
        self._build_review_transformer_layer()
        self._build_review_cl_project_head()
        self._build_atten()
        self._build_distReviewRepAtten()
        self._build_entity_review_cl_project_head()

    def _build_atten(self):
        self.ReviewHiddenStateAttenFunc = SelfAttentionSeq(self.config.token_emb_dim, self.config.token_emb_dim)

    def _build_review_transformer_layer(self):
        self.register_buffer('R_START', torch.tensor([self.config.start_token_idx], dtype=torch.long))

        self.review_encoder = TransformerEncoder(
            self.config.n_heads,
            self.config.n_layers,
            self.config.token_emb_dim,
            self.config.ffn_size,
            self.config.vocab_size,
            self.token_embedding,
            self.config.dropout,
            self.config.attention_dropout,
            self.config.relu_dropout,
            self.config.pad_token_idx,
            self.config.learn_positional_embeddings,
            self.config.embeddings_scale,
            self.config.reduction,
            self.config.n_positions
        )

        self.review_user_rep_dense = nn.Linear(self.config.token_emb_dim, self.config.user_emb_dim)

    def _build_review_cl_project_head(self):
        self.review_project_head_fc1 = nn.Linear(self.config.token_emb_dim, self.config.token_emb_dim)
        self.review_project_head_fc2 = nn.Linear(self.config.token_emb_dim, self.config.user_emb_dim)

    def _build_distReviewRepAtten(self):
        self.distReviewRepAtten = SelfAttentionBatch(self.config.token_emb_dim, self.config.token_emb_dim)

    def get_review_rep(self, batch, mode):
        review_user_rep, review_reps, review_state = self.get_review_user_rep_and_review_rep(batch, mode) # (bs, dim), (~bs*nb_review, dim)

        return review_user_rep # (bs, dim)

    def get_review_user_rep_and_review_rep(self, batch, mode):
        distInfo = batch['distInfo']  # (~bs*nb_review, seq_len)
        distInfo_mask = batch['distInfo_mask']  # (~bs*nb_review, seq_len)
        distInfo_pad_mask = batch['distInfo_pad_mask']  # (~bs*nb_review, seq_len)
        NbdistInfo = batch['NbdistInfo'] # [bs]

        review_user_rep, review_reps, review_state = self._get_review_user_rep(
            distInfo, distInfo_mask, distInfo_pad_mask, NbdistInfo) 
        # (bs, dim), (~bs*nb_review, dim), (~bs*nb_review, seq_len, dim)

        return review_user_rep, review_reps, review_state # (bs, dim), (~bs*nb_review, dim), (~bs*nb_review, seq_len, dim)

    def _get_review_user_rep(self, distInfo, review_mask, review_pad_mask, NbdistInfo):
        # (~bs*nb_review, seq_len), (~bs*nb_review, seq_len), (~bs*nb_review, seq_len), [bs]
        cls_state, review_state = self._get_hidden_state_review_transformer(distInfo)  # (~bs*nb_review, dim), (~bs*nb_review, seq_len, dim)
        review_reps = self.ReviewHiddenStateAttenFunc(review_state, review_pad_mask)  # (~bs*nb_review, dim)
        review_user_rep = self.encode_review_user_rep(review_reps, NbdistInfo) # (bs, dim)

        return review_user_rep, review_reps, review_state # (bs, dim), (~bs*nb_review, dim), (~bs*nb_review, seq_len, dim)
    
    def _get_hidden_state_review_transformer(self, info):
        # info = (bs, seq_len)
        state, mask = self.review_encoder(info)
        cls_state = state[:, 0, :] # (bs, dim)

        return cls_state, state # (bs, dim), 

    def _build_entity_review_cl_project_head(self):
        self.entity_review_project_head_fc1 = nn.Linear(self.config.token_emb_dim, self.config.token_emb_dim)
        self.entity_review_project_head_fc2 = nn.Linear(self.config.token_emb_dim, self.config.user_emb_dim)

    def get_project_review_rep(self, batch, mode):
        distInfo = batch['distInfo']  # (~bs*nb_review, seq_len)
        distInfo_mask = batch['distInfo_mask']  # (~bs*nb_review, seq_len)
        distInfo_pad_mask = batch['distInfo_pad_mask']  # (~bs*nb_review, seq_len)
        NbdistInfo = batch['NbdistInfo'] # [bs]

        review_user_rep, review_rep, review_state = self._get_review_user_rep(distInfo, distInfo_mask, distInfo_pad_mask, NbdistInfo) # (bs, dim), (~bs*nb_review, tok_dim)
        project_review_user_rep = self._project_review_user_rep(review_user_rep)  # (bs, dim)
        
        entities_has_distInfo = batch['entities_has_distInfo']
        if len(set(entities_has_distInfo)) == 1 and entities_has_distInfo[0] == -2:
            proj_entities_review_rep = torch.zeros((1, self.config.user_emb_dim)).to(self.config.device)
        else:
            proj_entities_review_rep = self._project_review_rep(review_rep) # (~bs*nb_review, user_dim)

        return project_review_user_rep, proj_entities_review_rep # (bs, dim), (~bs*nb_review, user_dim)
    
    def _project_review_rep(self, review_rep):
        review_rep = self.entity_review_project_head_fc1(review_rep)
        review_rep = F.relu(review_rep)
        review_rep = self.entity_review_project_head_fc2(review_rep)

        return review_rep

    def encode_review_user_rep(self, review_reps, NbdistInfo):
        # review_reps = (~bs*nb_review, dim)
        start_idx = 0

        review_user_rep = []
        for nbDistInfo in NbdistInfo:
            end_idx = start_idx + nbDistInfo

            if start_idx == end_idx:
                sample_review_reps = torch.zeros((1, self.config.token_emb_dim)).to(self.config.device)
            else:
                sample_review_reps = review_reps[start_idx: end_idx]  # (nb_review, dim)

            sample_review_user_rep = self.distReviewRepAtten(sample_review_reps) # (dim)
            review_user_rep.append(sample_review_user_rep.unsqueeze(0))

            start_idx += nbDistInfo
        review_user_rep = torch.cat(review_user_rep, dim=0) # (bs, dim)

        assert review_user_rep.shape[0] == len(NbdistInfo)
        return review_user_rep # (bs, dim)
    
    def _project_review_user_rep(self, review_user_rep):
        review_user_rep = self.review_project_head_fc1(review_user_rep) # (bs, dim)
        review_user_rep = F.relu(review_user_rep) # (bs, dim)
        review_user_rep = self.review_project_head_fc2(review_user_rep) # (bs, dim)

        return review_user_rep # (bs, dim)

    def get_review_sample_reps(self, batch, mode, review_reps, review_state):
        # review_reps, review_state = (~bs*nb_review, tok_dim), (~bs*nb_review, seq_len, dim)
        from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt
        NbdistInfo = batch['NbdistInfo'] # [bs], List[nb_review]
        distInfo_pad_mask = batch['distInfo_pad_mask']  # (~bs*nb_review, seq_len)
        _, seq_len = distInfo_pad_mask.shape
        bs = len(NbdistInfo)
        max_nb_review = max(NbdistInfo)
        nb_reviews = sum(NbdistInfo)
        assert nb_reviews == review_reps.shape[0]

        start_idx = 0
        review_pad_reps = [] # (bs, max_nb_review, tok_dim)
        review_pad_mask = [] # (bs, max_nb_review)
        review_token_padding_mask = [] # (bs, max_nb_review, seq_len)
        review_token_reps = [] # (bs, max_nb_review, seq_len)
        
        for nbDistInfo in NbdistInfo:
            end_idx = start_idx + nbDistInfo

            if start_idx == end_idx:
                sample_review_reps = torch.zeros((max_nb_review, self.config.token_emb_dim)).to(self.config.device) # (max_nb_review, tok_dim)
                sample_max_nb_review = torch.zeros((max_nb_review, seq_len)).to(self.config.device) # (max_nb_review, tok_dim)
                sample_review_token_reps = torch.zeros((max_nb_review, seq_len, self.config.token_emb_dim)).to(self.config.device) 
                # (max_nb_review, seq_len, dim)
            else:
                sample_review_reps = review_reps[start_idx: end_idx]  # (nb_review, tok_dim)
                nb_padding_review = max_nb_review - sample_review_reps.shape[0]
                padding_sample_review_reps = torch.zeros((nb_padding_review, self.config.token_emb_dim)).to(self.config.device)
                sample_review_reps = torch.cat([sample_review_reps, padding_sample_review_reps], dim=0)  # (max_nb_review, tok_dim)

                sample_max_nb_review = distInfo_pad_mask[start_idx: end_idx]  # (nb_review, tok_dim)
                padding_sample_max_nb_review = torch.zeros((nb_padding_review, seq_len)).to(self.config.device)
                sample_max_nb_review = torch.cat([sample_max_nb_review, padding_sample_max_nb_review], dim=0)  # (max_nb_review, tok_dim)

                sample_review_token_reps = review_state[start_idx: end_idx]  # (nb_review, seq_len, dim)
                padding_sample_review_token_reps = torch.zeros((nb_padding_review, seq_len, self.config.token_emb_dim)).to(self.config.device)
                sample_review_token_reps = torch.cat([sample_review_token_reps, padding_sample_review_token_reps], dim=0)
                # (max_nb_review, seq_len, dim)

            review_pad_reps.append(sample_review_reps)
            review_pad_mask.append([1]*nbDistInfo)
            review_token_padding_mask.append(sample_max_nb_review)
            review_token_reps.append(sample_review_token_reps)
            start_idx += nbDistInfo
        
        review_pad_reps = torch.stack(review_pad_reps)
        assert review_pad_reps.shape == (bs, max_nb_review, self.config.token_emb_dim)

        review_pad_mask = padded_tensor(items=review_pad_mask, pad_idx=0).to(self.config.device)
        assert review_pad_mask.shape == (bs, max_nb_review)

        review_token_padding_mask = torch.stack(review_token_padding_mask)
        assert review_token_padding_mask.shape == (bs, max_nb_review, seq_len)

        review_token_reps = torch.stack(review_token_reps)
        assert review_token_reps.shape == (bs, max_nb_review, seq_len, self.config.token_emb_dim)

        return review_pad_reps, review_pad_mask, review_token_reps, review_token_padding_mask
        # (bs, nb_review, tok_dim), (bs, nb_review), (bs, n_review, seq_len3)


class CoarseFineDRUserModel(nn.Module):
    def __init__(self, config, token_embedding):
        super().__init__()
        self.config = config
        self.token_embedding = token_embedding
        
        self._build_model()
        
    def _build_model(self):
        self.kGModel = CoarseFineKGModel(self.config)
        self.contextModel = CoarseFineContextModel(self.config, self.token_embedding)
        self.reviewModel = CoarseFineDistReviewModel(self.config, self.token_embedding)

    def get_project_user_rep(self, batch, mode):
        project_kg_user_rep, duplicate_removal_ekg_ids, duplicate_removal_ekg_conv_ids, project_entity_kg_reps = self.kGModel.get_project_kg_rep(
            batch, mode) # (bs, dim), [<=nb_ekg], [<=nb_ekg], (<=~bs*n_eic, user_dim) or (1, user_dim)
        project_context_user_rep, project_entity_context_text_reps = self.contextModel.get_project_context_rep(
            batch, mode) # (bs, dim), (~bs*n_eic, user_dim) or (1, user_dim)
        project_review_user_rep, proj_entities_review_rep = self.reviewModel.get_project_review_rep(
            batch, mode) # (bs, dim), (bs*nb_review, dim) or (1, user_dim)
        
        return project_kg_user_rep, project_context_user_rep, project_review_user_rep, \
            project_entity_kg_reps, project_entity_context_text_reps, proj_entities_review_rep, \
            duplicate_removal_ekg_ids, duplicate_removal_ekg_conv_ids
    
    def get_user_rep(self, batch, mode):
        user_rep, item_rep = self.get_kg_user_rep(batch, mode)

        return user_rep, item_rep

    def get_kg_user_rep(self, batch, mode):
        user_rep, kg_embedding = self.kGModel.get_kg_rep(batch, mode) # (bs, user_dim), (n_entities, user_dim)

        return user_rep, kg_embedding # (bs, user_dim), (n_entities, user_dim)