# @Time    :   2022/1/1
# @Author  :   Yuanhang Zhou
# @email   :   sdzyh002@gmail.com

r"""
TODO
TextCNN
=======
References:
    Kim, Yoon. `"Convolutional Neural Networks for Sentence Classification."`_ in EMNLP 2014.

.. _`"Convolutional Neural Networks for Sentence Classification."`:
   https://www.aclweb.org/anthology/D14-1181/

"""
import os
import ipdb
import math
import torch
import numpy as np
import torch.nn.functional as F
from loguru import logger
from torch import nn

from crslab.model.base import BaseModel
from crslab.model.utils.functions import edge_to_pyg_format

from crslab.model.crs.c2crs.muli_type_data_module import CoarseFineDRUserModel
from crslab.model.crs.c2crs.pre_training_module import CoarseToFinePretrainModel
from crslab.model.crs.c2crs.recommender_module import RecommenderModule
from crslab.model.crs.c2crs.conversation_module import CFSelectionConvModel


class ModelConfig(object):
    def __init__(self, opt, device, vocab, side_data):
        self._init_vocab(vocab, opt, side_data, device)
        self._init_transformer(vocab, opt, side_data, device)
        self._init_kg(vocab, opt, side_data, device)
        self._init_others(vocab, opt, side_data, device)
    
    def _init_vocab(self, vocab, opt, side_data, device):
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.vocab_size = vocab['vocab_size']
        self.token_emb_dim = opt.get('token_emb_dim', 300)
        self.pretrain_embedding = side_data.get('embedding', None)
    
    def _init_transformer(self, vocab, opt, side_data, device):
        self.n_heads = opt.get('n_heads', 2)
        self.n_layers = opt.get('n_layers', 2)
        self.ffn_size = opt.get('ffn_size', 300)
        self.dropout = opt.get('dropout', 0.1)
        self.attention_dropout = opt.get('attention_dropout', 0.0)
        self.relu_dropout = opt.get('relu_dropout', 0.1)
        self.embeddings_scale = opt.get('embedding_scale', True)
        self.learn_positional_embeddings = opt.get('learn_positional_embeddings', False)
        self.reduction = opt.get('reduction', False)
        self.n_positions = opt.get('n_positions', 1024)
        self.longest_label = opt.get('longest_label', 1)

    def _init_kg(self, vocab, opt, side_data, device):
        self.kg_name = opt['kg_name']
        self.n_entity = side_data[self.kg_name]['n_entity']
        entity_kg = side_data[self.kg_name]
        self.n_relation = entity_kg['n_relation']
        self.edge_idx, self.edge_type = edge_to_pyg_format(entity_kg['edge'], 'RGCN')
        self.edge_idx = self.edge_idx.to(device)
        self.edge_type = self.edge_type.to(device)
        self.num_bases = opt.get('num_bases', 8)
        self.kg_emb_dim = opt.get('kg_emb_dim', 300)
        self.user_emb_dim = self.kg_emb_dim

    def _init_others(self, vocab, opt, side_data, device):
        self.temperature = opt['temperature']
        self.device = device
        self.coarse_loss_lambda = opt.get('coarse_loss_lambda', 0.2)
        self.fine_loss_lambda = opt.get('fine_loss_lambda', 0.2)
        self.coarse_pretrain_epoch = opt.get('coarse_pretrain_epoch', 12)

class C2CRS_Model(BaseModel):
    def __init__(self, opt, device, vocab, side_data):
        self.config = ModelConfig(opt, device, vocab, side_data)
        self.vocab, self.side_data = vocab, side_data
        
        super(C2CRS_Model, self).__init__(opt, device)

        self._check_model()
    
    def _check_model(self):
        assert isinstance(self.user_model, CoarseFineDRUserModel)
        assert isinstance(self.pretrain_model, CoarseToFinePretrainModel)
        assert isinstance(self.recommender, RecommenderModule)
        assert isinstance(self.conv_model, CFSelectionConvModel)

    def build_model(self, *args, **kwargs):
        self._build_embedding()
        self._build_user_model()
        self._build_pretrain_model()
        self._build_recommender()
        self._build_conversation_model()
    
    def _build_embedding(self):
        if self.config.pretrain_embedding is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.config.pretrain_embedding, dtype=torch.float), 
                freeze=False,
                padding_idx=self.config.pad_token_idx)
        else:
            self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.token_emb_dim, self.config.pad_token_idx)
            nn.init.normal_(self.token_embedding.weight, mean=0, std=self.config.kg_emb_dim ** -0.5)
            nn.init.constant_(self.token_embedding.weight[self.config.pad_token_idx], 0)
        logger.debug('[Build embedding]')

    def _build_user_model(self):
        self.user_model = CoarseFineDRUserModel(self.config, self.token_embedding)
        logger.debug('[Build User_model]')
        
    def _build_pretrain_model(self):
        self.pretrain_model = CoarseToFinePretrainModel(self.config)
        logger.debug('[Build Pretrain_model]')
        
    def _build_recommender(self):
        self.recommender = RecommenderModule(self.config)
        logger.debug('[Build Recommender]')
        
    def _build_conversation_model(self):
        self._build_decoder_token_embedding()
        self.conv_model = CFSelectionConvModel(self.opt, self.device, self.vocab, self.side_data, self.decoder_token_embedding)
        logger.debug('[Build Conversation]')
    
    def _build_decoder_token_embedding(self):
        self.decoder_token_embedding = nn.Embedding.from_pretrained(
                self.token_embedding.weight,
                freeze=False,
                padding_idx=self.config.pad_token_idx)

    def freeze_parameters(self):
        freeze_models = self.build_freeze_models()
        logger.info('[freeze {} parameter unit]'.format(len(freeze_models)))

        for model in freeze_models:
            for p in model.parameters():
                p.requires_grad = False
    
    def build_freeze_models(self):
        freeze_models = []

        if 'k' in self.opt['freeze_parameters_name']:
            freeze_models.append(self.user_model.kGModel)
        if 'r' in self.opt['freeze_parameters_name']:
            freeze_models.append(self.user_model.reviewModel)
        if 'c' in self.opt['freeze_parameters_name']:
            freeze_models.append(self.user_model.contextModel)
        if 't' in self.opt['freeze_parameters_name']:
            freeze_models.append(self.token_embedding)

        return freeze_models

    def pretrain(self, batch, mode, epoch):
        # contrastive learning for pretraining
        loss = self.pretrain_model(self.user_model, batch, mode, epoch)

        return loss
    
    def recommend(self, batch, mode):
        rec_loss, rec_scores = self.recommender(self.user_model, batch, mode)

        return rec_loss, rec_scores

    def converse(self, batch, mode):
        loss, preds = self.conv_model(
            batch, 
            mode, 
            self.user_model.contextModel.context_encoder,
            self.user_model.kGModel,
            self.user_model.reviewModel
            )

        return loss, preds

 