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

class RecommenderModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self._build_model()
        
    def _build_model(self):
        self.rec_bias = nn.Linear(self.config.user_emb_dim, self.config.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()

    def forward(self, user_model, batch, mode):
        y = batch['movie_to_rec']

        user_rep, item_rep = user_model.get_user_rep(batch, mode) # (bs, user_dim), (n_entities, user_dim)
        rec_scores = F.linear(user_rep, item_rep, self.rec_bias.bias)  # (bs, n_entity)
        rec_loss = self.rec_loss(rec_scores, y)

        return rec_loss, rec_scores
