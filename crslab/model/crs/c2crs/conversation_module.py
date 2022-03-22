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

from crslab.model.utils.modules.transformer import MultiHeadAttention, TransformerFFN, _create_selfattn_mask, \
    _normalize, \
    create_position_codes


NEAR_INF_FP16 = 65504
NEAR_INF = 1e20
def neginf(dtype):
    """Returns a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF
    
class DBModel(nn.Module):
    def __init__(self, opt, device, vocab, side_data):
        super().__init__()
        # vocab
        self.vocab_size = vocab['vocab_size']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.token_emb_dim = opt['token_emb_dim']
        self.pretrained_embedding = side_data.get('embedding', None)
        # kg
        self.n_word = side_data['word_kg']['n_entity']
        self.kg_name = opt['kg_name']
        self.n_entity = side_data[self.kg_name]['n_entity']
        self.pad_word_idx = vocab['pad_word']
        self.pad_entity_idx = vocab['pad_entity']
        entity_kg = side_data['entity_kg']
        self.n_relation = entity_kg['n_relation']
        entity_edges = entity_kg['edge']
        self.entity_edge_idx, self.entity_edge_type = edge_to_pyg_format(entity_edges, 'RGCN')
        self.entity_edge_idx = self.entity_edge_idx.to(device)
        self.entity_edge_type = self.entity_edge_type.to(device)
        word_edges = side_data['word_kg']['edge']
        self.word_edges = edge_to_pyg_format(word_edges, 'GCN').to(device)
        self.num_bases = opt['num_bases']
        self.kg_emb_dim = opt['kg_emb_dim']
        # transformer
        self.n_heads = opt['n_heads']
        self.n_layers = opt['n_layers']
        self.ffn_size = opt['ffn_size']
        self.dropout = opt['dropout']
        self.attention_dropout = opt['attention_dropout']
        self.relu_dropout = opt['relu_dropout']
        self.learn_positional_embeddings = opt['learn_positional_embeddings']
        self.embeddings_scale = opt['embeddings_scale']
        self.reduction = opt['reduction']
        self.n_positions = opt['n_positions']
        self.response_truncate = opt.get('response_truncate', 20)

        self._build_model()
        
    def _build_model(self):
        self._build_conversation_layer()

    def _build_conversation_layer(self):
        self.conv_entity_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)
        self.conv_entity_attn_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)

    def forward(self, batch, mode, kgModel):
        entity_attn_rep, entity_representations = self.entity_model_kbrd(batch, mode, kgModel) # (bs, dim), (bs, n_context_entities, dim)
        conv_entity_emb, conv_entity_reps = self.conv_entaity_model(entity_attn_rep, entity_representations) # (bs, ffn_size), (bs, n_context_entities, ffn_size)

        return entity_attn_rep, entity_representations, conv_entity_emb, conv_entity_reps 
        # (bs, dim), (bs, n_context_entities, dim), (bs, ffn_size), (bs, n_context_entities, ffn_size)

    def entity_model_kbrd(self, batch, mode, kgModel):
        context_entities_kbrd = batch['context_entities_kbrd']  # [bs, nb_context_entities]
        context_entities = batch['context_entities']  # (bs, entity_truncate)

        user_rep, kg_embedding = kgModel._get_kg_user_rep(context_entities_kbrd) # (bs, dim), (n_entities, dim)
        entity_representations = kg_embedding[context_entities] # (bs, entity_truncate, dim)

        return user_rep, entity_representations
    
    def conv_entaity_model(self, entity_attn_rep, entity_representations):
        # encoder-decoder
        conv_entity_emb = self.conv_entity_attn_norm(entity_attn_rep) # (bs, ffn_size)
        conv_entity_reps = self.conv_entity_norm(entity_representations) # (bs, n_context_entities, ffn_size)

        return conv_entity_emb, conv_entity_reps # (bs, ffn_size), (bs, n_context_entities, ffn_size)


class CoarseReviewModelForDecoder(nn.Module):
    def __init__(self, opt, device, vocab, side_data):
        super().__init__()
        # vocab
        self.vocab_size = vocab['vocab_size']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.token_emb_dim = opt['token_emb_dim']
        self.pretrained_embedding = side_data.get('embedding', None)
        # kg
        self.n_word = side_data['word_kg']['n_entity']
        self.kg_name = opt['kg_name']
        self.n_entity = side_data[self.kg_name]['n_entity']
        self.pad_word_idx = vocab['pad_word']
        self.pad_entity_idx = vocab['pad_entity']
        entity_kg = side_data['entity_kg']
        self.n_relation = entity_kg['n_relation']
        entity_edges = entity_kg['edge']
        self.entity_edge_idx, self.entity_edge_type = edge_to_pyg_format(entity_edges, 'RGCN')
        self.entity_edge_idx = self.entity_edge_idx.to(device)
        self.entity_edge_type = self.entity_edge_type.to(device)
        word_edges = side_data['word_kg']['edge']
        self.word_edges = edge_to_pyg_format(word_edges, 'GCN').to(device)
        self.num_bases = opt['num_bases']
        self.kg_emb_dim = opt['kg_emb_dim']
        # transformer
        self.n_heads = opt['n_heads']
        self.n_layers = opt['n_layers']
        self.ffn_size = opt['ffn_size']
        self.dropout = opt['dropout']
        self.attention_dropout = opt['attention_dropout']
        self.relu_dropout = opt['relu_dropout']
        self.learn_positional_embeddings = opt['learn_positional_embeddings']
        self.embeddings_scale = opt['embeddings_scale']
        self.reduction = opt['reduction']
        self.n_positions = opt['n_positions']
        self.response_truncate = opt.get('response_truncate', 20)

        self._build_model()
        
    def _build_model(self):
        self._build_conv_concept_encoder()

    def _build_conv_concept_encoder(self):
        self.conv_review_attn_norm = nn.Linear(self.token_emb_dim, self.ffn_size)
        self.conv_review_norm = nn.Linear(self.token_emb_dim, self.ffn_size)
    
    def forward(self, batch, mode, reviewModel):
        review_user_rep, review_reps, review_pad_reps, review_padding_mask, review_token_reps, review_token_padding_mask = \
            self.model_review(batch, mode, reviewModel)
        # (bs, dim), (~bs*nb_review, dim), (bs, n_review, dim), (bs, nb_review), (bs, n_review, seq_len3, dim), (bs, n_review, seq_len3)

        conv_review_emb, conv_review_reps = self.conv_review_model(review_user_rep, review_pad_reps) 
        # (bs, ffn_size), (bs, n_review, ffn_size)

        return conv_review_emb, conv_review_reps, review_padding_mask, review_token_reps, review_token_padding_mask
        # (bs, ffn_size), (bs, n_review, dim), (bs, ffn_size), (bs, n_review, seq_len3, dim), (bs, n_review, seq_len3)
    
    def model_review(self, batch, mode, reviewModel):
        review_user_rep, review_reps, review_state = reviewModel.get_review_user_rep_and_review_rep(batch, mode) 
        # (bs, dim), (~bs*nb_review, dim), (~bs*nb_review, seq_len3, dim)
        review_pad_reps, review_padding_mask, review_token_reps, review_token_padding_mask = reviewModel.get_review_sample_reps(
            batch, mode, review_reps, review_state) 
        # (bs, nb_review, dim), (bs, nb_review), (bs, n_review, seq_len3, dim), (bs, n_review, seq_len3)

        return review_user_rep, review_reps, review_pad_reps, \
            review_padding_mask, review_token_reps, review_token_padding_mask
        # (bs, dim), (~bs*nb_review, dim), (bs, n_review, dim), 
        # (bs, nb_review), (bs, n_review, seq_len3, dim), (bs, n_review, seq_len3)

    def conv_review_model(self, review_attn_rep, review_representations):
        # (bs, dim), (bs, n_review, dim), (bs, nb_review, dim), (bs, seq_len3, dim)
        conv_review_emb = self.conv_review_attn_norm(review_attn_rep) # (bs, ffn_size)
        conv_review_reps = self.conv_review_norm(review_representations) # (bs, n_context_words, ffn_size)

        return conv_review_emb, conv_review_reps
        # (bs, ffn_size), (bs, n_review, ffn_size)


class FineReviewDecoderAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=.0):
        super(FineReviewDecoderAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.dim_per_head = self.dim // self.n_heads

        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        # TODO: merge for the initialization step
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        # and set biases to 0
        self.out_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.out_lin.weight)

        self.fine_review_level_self_atten = SelfAttentionSeq(self.dim_per_head, self.dim_per_head)

    def forward(self, query, key=None, value=None, mask=None, mask2=None):
        # query: (bs, query_len, ffn_size)
        # key/value: (bs, nb_review, key_len, ffn_size)
        # mask: (bs, nb_review)
        # mask2: (bs, nb_review, key_len)
        query, key, value = self.set_q_k_v(query, key, value)
        bs, query_len, n_heads, dim_per_head, scale, key_len, dim, nb_review = self.set_hyper_parameters(query, key, mask, mask2)
        q, k, v = self.prepare_heads(query, key, value, bs, n_heads, dim_per_head)
        # q: (bs*n_heads, query_len, dim_per_head)
        # k/v: (bs*n_heads, nb_review, key_len, dim_per_head)
        out = self.compute_func(q, k, v, query, mask, mask2, bs, query_len, n_heads, dim_per_head, scale, key_len, dim, nb_review) # (bs, query_len, dim)

        return out

    def set_q_k_v(self, query, key, value):
        return query, key, value

    def set_hyper_parameters(self, query, key, mask, mask2):
        bs, query_len, dim = query.size()
        assert dim == self.dim, \
            f'Dimensions do not match: {dim} query vs {self.dim} configured'
        assert mask is not None, 'Mask is None, please specify a mask'
        assert mask2 is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)
        _, nb_review, key_len, dim = key.size()

        return bs, query_len, n_heads, dim_per_head, scale, key_len, dim, nb_review

    def prepare_heads(self, query, key, value, bs, n_heads, dim_per_head):
        # query: (bs, query_len, ffn_size)
        # key/value: (bs, nb_review, key_len, ffn_size)
        q = self.prepare_head_q(self.q_lin(query), bs, n_heads, dim_per_head) # (bs*n_heads, query_len, dim_per_head)
        k = self.prepare_head_kv(self.k_lin(key), bs, n_heads, dim_per_head) # (bs*n_heads, nb_review, key_len, dim_per_head)
        v = self.prepare_head_kv(self.v_lin(value), bs, n_heads, dim_per_head) # (bs*n_heads, nb_review, key_len, dim_per_head)

        return q, k, v

    def prepare_head_q(self, tensor, bs, n_heads, dim_per_head):
        # input is (bs, query_len, ffn_size)
        # output is (bs*n_heads, query_len, dim_per_head)
        bs, seq_len, _ = tensor.size()
        tensor = tensor.view(bs, tensor.size(1), n_heads, dim_per_head)
        tensor = tensor.transpose(1, 2).contiguous().view(
            bs*n_heads,
            seq_len,
            dim_per_head
        )
        return tensor
    
    def prepare_head_kv(self, tensor, bs, n_heads, dim_per_head):
        # input is (bs, nb_review, key_len, ffn_size)
        # output is (bs*n_heads, nb_review, key_len, dim_per_head)
        bs, nb_review, seq_len, _ = tensor.size()
        tensor = tensor.view(bs, nb_review, seq_len, n_heads, dim_per_head)
        tensor = tensor.transpose(1, 3).transpose(2, 3).contiguous().view(
            bs*n_heads,
            nb_review,
            seq_len,
            dim_per_head
        )
        return tensor

    def compute_func(self, q, k, v, query, mask, mask2, bs, query_len, n_heads, dim_per_head, scale, key_len, dim, nb_review):
        # q: (bs*n_heads, query_len, dim_per_head)
        # k/v: (bs*n_heads, nb_review, key_len, dim_per_head)
        # mask: (bs, nb_review)
        # mask2: (bs, nb_review, key_len)

        attentioned = self.token_level_atten(q, k, v, query, mask, mask2, bs, query_len, n_heads, dim_per_head, scale, key_len, dim, nb_review) 
        # (bs*n_heads*nb_review, query_len, dim_per_head)
        attentioned = self.review_level_atten(attentioned, mask, bs, n_heads, nb_review, query_len, dim_per_head) # (bs*n_heads, query_len, dim_per_head)
        attentioned = (
                attentioned.type_as(query)
                    .view(bs, n_heads, query_len, dim_per_head)
                    .transpose(1, 2).contiguous()
                    .view(bs, query_len, dim)
            )
        # (bs, query_len, dim)
        out = self.out_lin(attentioned)  # (bs, query_len, dim)

        return out  # (bs, query_len, dim)

    def get_attn_mask(self, mask, bs, key_len, n_heads, query_len):
        # Mask is [bs, key_len] (selfattn) or [bs, key_len, key_len] (enc attn)
        attn_mask = (
            (mask == 0)
                .view(bs, 1, -1, key_len)
                .repeat(1, n_heads, 1, 1)
                .expand(bs, n_heads, query_len, key_len)
                .view(bs*n_heads, query_len, key_len)
        )

        return attn_mask # (bs*n_heads, query_len, key_len)
   
    def token_level_atten(self, q, k, v, query, mask, mask2, bs, query_len, n_heads, dim_per_head, scale, key_len, dim, nb_review):
        # q: (bs*n_heads, query_len, dim_per_head)
        # k/v: (bs*n_heads, nb_review, key_len, dim_per_head)
        # query: (bs, seq_len2, ffn_size)
        # mask: (bs, nb_review)
        # mask2: (bs, nb_review, key_len)
        q = (q.unsqueeze(1)
            .expand(bs*n_heads, nb_review, query_len, dim_per_head)
            .reshape(bs*n_heads*nb_review, query_len, dim_per_head))
        k = k.view(bs*n_heads*nb_review, key_len, dim_per_head)
        dot_prod = q.div_(scale).bmm(k.transpose(-2, -1)) # (bs*n_heads*nb_review, query_len, key_len)
    
        attn_mask = self.get_token_level_attn_mask(mask2, bs, key_len, n_heads, query_len, nb_review) 
        # (bs*n_heads*nb_review, query_len, key_len)
        assert attn_mask.shape == dot_prod.shape

        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype)) # (bs*n_heads*nb_review, query_len, key_len)

        attn_weights = F.softmax(dot_prod, dim=-1).type_as(query) # (bs*n_heads*nb_review, query_len, key_len)
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        v = v.view(bs*n_heads*nb_review, key_len, dim_per_head)
        attentioned = attn_weights.bmm(v) # (bs*n_heads*nb_review, query_len, dim_per_head)

        return attentioned # (bs*n_heads*nb_review, query_len, dim_per_head)

    def review_level_atten(self, attentioned, mask, bs, n_heads, nb_review, query_len, dim_per_head):
        # self-attention or (bs, nb_review, dim) as query
        # attentioned: (bs*n_heads*nb_review, query_len, dim_per_head)
        # mask: (bs, nb_review) :the padding posistion should be 1
        attentioned = (attentioned
            .view(bs*n_heads, nb_review, query_len, dim_per_head)
            .transpose(1, 2).contiguous()
            .view(bs*n_heads*query_len, nb_review, dim_per_head)
            )
        # (bs*n_heads*query_len, nb_review, dim_per_head)
        mask = (mask
            .unsqueeze(1).unsqueeze(1)
            .expand(bs, n_heads, query_len, nb_review).contiguous()
            .view(bs*n_heads*query_len, nb_review)
        )
        assert attentioned.shape[:2] == mask.shape[:2]
        attentioned = self.fine_review_level_self_atten(attentioned, mask) # (bs*n_heads*query_len, dim_per_head)
        attentioned = attentioned.view(bs*n_heads, query_len, dim_per_head)

        return attentioned # (bs*n_heads, query_len, dim_per_head)

    def get_token_level_attn_mask(self, mask2, bs, key_len, n_heads, query_len, nb_review):
        # mask2: (bs, nb_review, key_len)
        attn_mask = (
            (mask2 == 0)
                .view(bs, 1, nb_review, 1, key_len)
                .repeat(1, n_heads, 1, 1, 1)
                .expand(bs, n_heads, nb_review, query_len, key_len)
                .view(bs*n_heads*nb_review, query_len, key_len)
        )

        return attn_mask # (bs*n_heads*nb_review, query_len, key_len)


class TransformerDecoderLayerCoarse(nn.Module):
    def __init__(
            self,
            n_heads,
            embedding_size,
            ffn_size,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size)

        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2 = nn.LayerNorm(embedding_size)

        self.encoder_db_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2_db = nn.LayerNorm(embedding_size)

        self.encoder_review_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2_review = nn.LayerNorm(embedding_size)

        self.fine_encoder_review_attention = FineReviewDecoderAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2_review2 = nn.LayerNorm(embedding_size)

        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, 
                inputs, 
                encoder_output, encoder_mask,
                conv_entity_reps, entity_padding_mask, 
                conv_review_reps, review_padding_mask,
                review_token_reps, review_token_padding_mask):
        '''
        input: (bs, seq_len2, dim)
        encoder_output, encoder_mask: (bs, seq_len, dim), (bs, seq_len)
        conv_entity_reps, entity_padding_mask: (bs, n_context_entities, ffn_size), (bs, entity_len)
        conv_review_reps, review_padding_mask: (bs, nb_review, ffn_size), (bs, nb_review)
        review_token_reps, review_token_padding_mask: (bs, nb_review, ffn_size), (bs, nb_review, seq_len3)
        '''
        inputs = self._decoder_self_attention(inputs)
        inputs = self._db_decode_cross_attention(inputs, conv_entity_reps, entity_padding_mask)
        inputs = self._coarse_review_decode_cross_attention(inputs, conv_review_reps, review_padding_mask)
        inputs = self._fine_review_decode_cross_attention(inputs, review_token_reps, review_padding_mask, review_token_padding_mask)
        # inputs = self._review_decode_cross_attention3(inputs, review_token_decode_atten_rep, review_token_padding_mask)
        inputs = self._context_decode_cross_attention(inputs, encoder_output, encoder_mask)
        inputs = self._ffn(inputs)
        
        return inputs # (bs, seq_len2, dim)

    def _decoder_self_attention(self, x):
        decoder_mask = _create_selfattn_mask(x)
        # first self attn
        residual = x
        # don't peak into the future!
        x = self.self_attention(query=x, mask=decoder_mask)
        x = self.dropout(x)  # --dropout
        x = x + residual
        x = _normalize(x, self.norm1)
        
        return x # (bs, seq_len2, dim)

    def _db_decode_cross_attention(self, x, conv_entity_reps, entity_padding_mask):
        # x: (bs, seq_len2, dim)
        # conv_entity_reps, entity_padding_mask: (bs, n_context_entities, ffn_size), (bs, entity_len)
        residual = x
        x = self.encoder_db_attention(
            query=x,
            key=conv_entity_reps,
            value=conv_entity_reps,
            mask=entity_padding_mask
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2_db)

        return x # (bs, seq_len2, dim)

    def _coarse_review_decode_cross_attention(self, x, conv_review_reps, review_padding_mask):
        # x: (bs, seq_len2, dim)
        # conv_review_reps, review_padding_mask: (bs, nb_review, ffn_size), (bs, nb_review)
        residual = x
        x = self.encoder_review_attention(
            query=x,
            key=conv_review_reps,
            value=conv_review_reps,
            mask=review_padding_mask
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2_review)

        return x # (bs, seq_len2, dim)

    def _fine_review_decode_cross_attention(self, x, review_token_reps, review_padding_mask, review_token_padding_mask):
        # x: (bs, seq_len2, dim)
        # (bs, nb_review, seq_len3, ffn_size), (bs, nb_review), (bs, nb_review, seq_len3)
        residual = x
        x = self.fine_encoder_review_attention(
            query=x,
            key=review_token_reps,
            value=review_token_reps,
            mask=review_padding_mask,
            mask2=review_token_padding_mask,
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2_review2)

        return x # (bs, seq_len2, dim)
        
    def _context_decode_cross_attention(self, x, encoder_output, encoder_mask):
        residual = x
        x = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2)

        return x # (bs, seq_len2, dim)

    def _ffn(self, x):
        # finally the ffn
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm3)

        return x # (bs, seq_len2, dim)


class TransformerDecoderLayerSelection(TransformerDecoderLayerCoarse):
    def __init__(
            self,
            opt,
            n_heads,
            embedding_size,
            ffn_size,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
    ):
        self.opt = opt
        super().__init__(
            n_heads,
            embedding_size,
            ffn_size,
            attention_dropout,
            relu_dropout,
            dropout)

    def forward(self, 
                inputs, 
                encoder_output, encoder_mask,
                conv_entity_reps, entity_padding_mask, 
                conv_review_reps, review_padding_mask,
                review_token_reps, review_token_padding_mask):
        '''
        input: (bs, seq_len2, dim)
        encoder_output, encoder_mask: (bs, seq_len, dim), (bs, seq_len)
        conv_entity_reps, entity_padding_mask: (bs, n_context_entities, ffn_size), (bs, entity_len)
        conv_review_reps, review_padding_mask: (bs, nb_review, ffn_size), (bs, nb_review)
        review_token_reps, review_token_padding_mask: (bs, nb_review, ffn_size), (bs, nb_review, seq_len3)
        '''

        inputs = self.forward_d_c_db_r_f(
                inputs, 
                encoder_output, encoder_mask,
                conv_entity_reps, entity_padding_mask, 
                conv_review_reps, review_padding_mask,
                review_token_reps, review_token_padding_mask)
        
        return inputs # (bs, seq_len2, dim)

    def forward_d_c_db_r_f(self, 
            inputs, 
            encoder_output, encoder_mask,
            conv_entity_reps, entity_padding_mask, 
            conv_review_reps, review_padding_mask,
            review_token_reps, review_token_padding_mask):
        logger.debug('[forward_d_c_db_r_f]')
        inputs = self._decoder_self_attention(inputs)
        inputs = self._context_decode_cross_attention(inputs, encoder_output, encoder_mask)
        inputs = self._db_decode_cross_attention(inputs, conv_entity_reps, entity_padding_mask)
        inputs = self._coarse_review_decode_cross_attention(inputs, conv_review_reps, review_padding_mask)
        inputs = self._ffn(inputs)

        return inputs


class TransformerDecoderKGCoarse(nn.Module):
    
    def __init__(
            self,
            n_heads,
            n_layers,
            embedding_size,
            ffn_size,
            vocabulary_size,
            embedding,
            dropout=0.0,
            attention_dropout=0.0,
            relu_dropout=0.0,
            embeddings_scale=True,
            learn_positional_embeddings=False,
            padding_idx=None,
            n_positions=1024,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(dropout)  # --dropout
        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding
        self.position_embeddings = self._init_postision_embeddings(n_positions, embedding_size, learn_positional_embeddings)
        self.layers = self._build_layers(n_heads, embedding_size, ffn_size, attention_dropout, relu_dropout, dropout)
    
    def _init_postision_embeddings(self, n_positions, embedding_size, learn_positional_embeddings):
        # create the positional embeddings
        position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=position_embeddings.weight
            )
        else:
            nn.init.normal_(position_embeddings.weight, 0, embedding_size ** -0.5)
        
        return position_embeddings
        
    def _build_layers(self, n_heads, embedding_size, ffn_size, attention_dropout, relu_dropout, dropout):
        layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layers.append(TransformerDecoderLayerCoarse(
                n_heads, embedding_size, ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
            ))

        return layers

    def forward(self, 
                inputs, 
                encoder_output, encoder_mask,
                conv_entity_reps, entity_padding_mask, 
                conv_review_reps, review_padding_mask,
                review_token_reps, review_token_padding_mask,
                incr_state=None):
        '''
        input: (bs, seq_len2, dim)
        encoder_output, encoder_mask: (bs, seq_len, dim), (bs, seq_len)
        conv_entity_reps, entity_padding_mask: (bs, n_context_entities, ffn_size), (bs, entity_len)
        conv_review_reps, review_padding_mask: (bs, nb_review, ffn_size), (bs, nb_review)
        review_token_reps, review_token_padding_mask: (bs, nb_review, ffn_size), (bs, nb_review, seq_len3)
        '''
        inputs = self.embed_input(inputs)  # (bs, seq_len2, dim)

        for layer in self.layers:
            inputs = layer(
                inputs, 
                encoder_output, encoder_mask,
                conv_entity_reps, entity_padding_mask, 
                conv_review_reps, review_padding_mask,
                review_token_reps, review_token_padding_mask) # (bs, seq_len2, dim)

        return inputs, None # (bs, seq_len, embed_dim)

    def embed_input(self, input):
        tensor = self.embeddings(input)  # (bs, seq_len, embed_dim)

        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        
        positions_embedding = self.get_postition_embeddings(input, tensor)
        tensor = tensor + positions_embedding
        
        tensor = self.dropout(tensor)  # --dropout

        return tensor
    
    def get_postition_embeddings(self, input, tensor):
        seq_len = input.size(1)
        positions = input.new(seq_len).long()  # (seq_len)
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)  # (1, seq_len)
        positions_embedding = self.position_embeddings(positions).expand_as(tensor)

        return positions_embedding


class TransformerDecoderKGSelection(TransformerDecoderKGCoarse):
    
    def __init__(
            self,
            opt,
            n_heads,
            n_layers,
            embedding_size,
            ffn_size,
            vocabulary_size,
            embedding,
            dropout=0.0,
            attention_dropout=0.0,
            relu_dropout=0.0,
            embeddings_scale=True,
            learn_positional_embeddings=False,
            padding_idx=None,
            n_positions=1024,
    ):
        self.opt = opt
        super().__init__(
            n_heads,
            n_layers,
            embedding_size,
            ffn_size,
            vocabulary_size,
            embedding,
            dropout,
            attention_dropout,
            relu_dropout,
            embeddings_scale,
            learn_positional_embeddings,
            padding_idx,
            n_positions)

    def _build_layers(self, n_heads, embedding_size, ffn_size, attention_dropout, relu_dropout, dropout):
        layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layers.append(TransformerDecoderLayerSelection(
                self.opt,
                n_heads, embedding_size, ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
            ))

        return layers


class DecoderCNSelectionModel(nn.Module):
    def __init__(self, opt, device, vocab, side_data, decoder_token_embedding):
        super().__init__()
        self.opt, self.device, self.vocab, self.side_data = opt, device, vocab, side_data
        # vocab
        self.vocab_size = vocab['vocab_size']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.token_emb_dim = opt['token_emb_dim']
        self.pretrained_embedding = side_data.get('embedding', None)
        # kg
        self.n_word = side_data['word_kg']['n_entity']
        self.kg_name = opt['kg_name']
        self.n_entity = side_data[self.kg_name]['n_entity']
        self.pad_word_idx = vocab['pad_word']
        self.pad_entity_idx = vocab['pad_entity']
        entity_kg = side_data['entity_kg']
        self.n_relation = entity_kg['n_relation']
        entity_edges = entity_kg['edge']
        self.entity_edge_idx, self.entity_edge_type = edge_to_pyg_format(entity_edges, 'RGCN')
        self.entity_edge_idx = self.entity_edge_idx.to(device)
        self.entity_edge_type = self.entity_edge_type.to(device)
        word_edges = side_data['word_kg']['edge']
        self.word_edges = edge_to_pyg_format(word_edges, 'GCN').to(device)
        self.num_bases = opt['num_bases']
        self.kg_emb_dim = opt['kg_emb_dim']
        # transformer
        self.n_heads = opt['n_heads']
        self.n_layers = opt['n_layers']
        self.ffn_size = opt['ffn_size']
        self.dropout = opt['dropout']
        self.attention_dropout = opt['attention_dropout']
        self.relu_dropout = opt['relu_dropout']
        self.learn_positional_embeddings = opt['learn_positional_embeddings']
        self.embeddings_scale = opt['embeddings_scale']
        self.reduction = opt['reduction']
        self.n_positions = opt['n_positions']
        self.response_truncate = opt.get('response_truncate', 20)
        self.decoder_token_embedding = decoder_token_embedding

        self.decoder_token_prob_weight = side_data.get('decoder_token_prob_weight', None)
        if self.decoder_token_prob_weight is not None:
            self.decoder_token_prob_weight = self.decoder_token_prob_weight.to(self.device)
        self.is_weight_logits = opt.get('is_weight_logits', False)

        self.is_coarse_weight_loss = opt.get('is_coarse_weight_loss', False)
        # assert not(self.is_weight_logits and self.is_coarse_weight_loss)

        self._build_model()
        
    def _build_model(self):
        self.register_buffer('START', torch.tensor([self.start_token_idx], dtype=torch.long))

        self.conv_decoder = TransformerDecoderKGSelection(
            self.opt,
            self.n_heads, self.n_layers, self.token_emb_dim, self.ffn_size, self.vocab_size,
            embedding=self.decoder_token_embedding,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=self.relu_dropout,
            embeddings_scale=self.embeddings_scale,
            learn_positional_embeddings=self.learn_positional_embeddings,
            padding_idx=self.pad_token_idx,
            n_positions=self.n_positions
        )

        self.conv_loss = self.build_loss_func()

        self._build_copy_network()
    
    def build_loss_func(self):
        if self.is_coarse_weight_loss:
            conv_loss = Handle_Croess_Entropy_Loss(ignore_index=self.pad_token_idx, weight=self.decoder_token_prob_weight.squeeze())
            # conv_loss = coarse_weight_loss(ignore_index=self.pad_token_idx)
        else:
            conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)
        
        return conv_loss

    def _build_copy_network(self):
        n_copy_source = 3 if '3' in self.opt['logit_type'] else 2
        self.copy_norm = nn.Linear(self.ffn_size * n_copy_source, self.token_emb_dim)
        self.copy_output = nn.Linear(self.token_emb_dim, self.vocab_size)

        self.fusion_latent_norm = nn.Linear(self.ffn_size * n_copy_source, self.token_emb_dim)

    def forward(self, 
                mode, 
                encoder_output, encoder_mask, 
                conv_entity_emb, conv_entity_reps, entity_padding_mask,
                conv_review_emb, conv_review_reps, review_padding_mask,
                review_token_reps, review_token_padding_mask,
                response):
        '''
        encoder_output, encoder_mask: (bs, seq_len, dim), (bs, seq_len)
        conv_entity_reps, entity_padding_mask: (bs, n_context_entities, ffn_size), (bs, entity_len)
        conv_review_reps, review_padding_mask: (bs, nb_review, ffn_size), (bs, nb_review)
        review_token_reps, review_token_padding_mask: (bs, nb_review, ffn_size), (bs, nb_review, seq_len3)
        response: (bs, seq_len)
        '''

        mode2decode_func = {
            'train': self._decode_forced_with_kg,
            'val': self._decode_forced_with_kg,
            'test': self._decode_greedy_with_kg
        }
        decode_func = mode2decode_func[mode]

        logits, preds, loss = decode_func(
            encoder_output, encoder_mask, 
            conv_entity_emb, conv_entity_reps, entity_padding_mask,
            conv_review_emb, conv_review_reps, review_padding_mask,
            review_token_reps, review_token_padding_mask,
            response)

        return loss, preds

    def _starts(self, bs):
        """Return bs start tokens."""
        return self.START.detach().expand(bs, 1)

    def _decode_forced_with_kg(
            self, 
            encoder_output, encoder_mask, 
            conv_entity_emb, conv_entity_reps, entity_padding_mask,
            conv_review_emb, conv_review_reps, review_padding_mask,
            review_token_reps, review_token_padding_mask,
            response):
        batch_size, seq_len = response.shape
        start = self._starts(batch_size)
        inputs = torch.cat((start, response[:, :-1]), dim=-1).long() # (bs, seq_len)
        # inputs = response[:, :] # (bs, seq_len)

        dialog_latent, _ = self.conv_decoder(
            inputs, 
            encoder_output, encoder_mask,
            conv_entity_reps, entity_padding_mask, 
            conv_review_reps, review_padding_mask,
            review_token_reps, review_token_padding_mask)  
        # (bs, seq_len, dim)

        gen_logits, preds, loss = self._force_process_dialog_latent(conv_entity_emb, conv_review_emb, dialog_latent, response, conv_review_reps, review_padding_mask)
        return gen_logits, preds, loss
    
    def _force_process_dialog_latent(self, entity_latent, review_latent, dialog_latent, response, conv_review_reps, review_padding_mask):
        # (bs, dim), (bs, dim), (bs, seq_len1, dim), (bs, seq_len), (bs, nb_review, ffn_size), (bs, nb_review)
        batch_size, seq_len = response.shape
        entity_latent = entity_latent.unsqueeze(1).expand(-1, seq_len, -1) # (bs, seq_len, ffn_size)
        review_latent = review_latent.unsqueeze(1).expand(-1, seq_len, -1) # (bs, seq_len, ffn_size)

        logits = self._get_logits(entity_latent, review_latent, dialog_latent, conv_review_reps, review_padding_mask)  # (bs, seq_len, vocab_size)
        preds = logits.argmax(dim=-1)
        loss = self._force_get_gen_loss(logits, response)

        return logits, preds, loss

    def _force_get_gen_loss(self, logits, response):
        # (bs, seq_len, vocab_size), (bs, seq_len)
        logits = logits.view(-1, logits.shape[-1]) # (bs*seq_len, nb_tok)
        response = response.view(-1) # (bs*seq_len)
        
        # n = 2
        # loss = self.conv_loss(logits[:n], response[:n])
        # logger.info(f'{logits[:n]}')
        # logger.info(f'{response[:n]}')
        # logger.info(f'{loss}')
        # ipdb.set_trace()

        loss = self.conv_loss(logits, response)

        return loss

    def _decode_greedy_with_kg(
            self, 
            encoder_output, encoder_mask, 
            conv_entity_emb, conv_entity_reps, entity_padding_mask,
            conv_review_emb, conv_review_reps, review_padding_mask,
            review_token_reps, review_token_padding_mask,
            response):
        bs = encoder_output.shape[0]
        inputs = self._starts(bs).long() # (bs, 1)
        incr_state = None
        logits = []

        for _ in range(self.response_truncate):
            dialog_latent, incr_state = self.conv_decoder(
                inputs, 
                encoder_output, encoder_mask,
                conv_entity_reps, entity_padding_mask, 
                conv_review_reps, review_padding_mask,
                review_token_reps, review_token_padding_mask,
                incr_state)  
            # (bs, seq_len, dim), None

            cur_time_logits, preds = self._greedy_process_dialog_latent(conv_entity_emb, conv_review_emb, dialog_latent, conv_review_reps, review_padding_mask)
            logits.append(cur_time_logits)
            inputs = torch.cat((inputs, preds), dim=1) # (bs, gen_response_len)

            finished = ((inputs == self.end_token_idx).sum(dim=-1) > 0).sum().item() == bs
            if finished:
                break

        logits = torch.cat(logits, dim=1) # (bs, response_truncate, nb_tok)
        loss = None

        return logits, inputs, loss # (bs, response_truncate, nb_tok), 

    def _greedy_process_dialog_latent(self, entity_latent, review_latent, dialog_latent, conv_review_reps, review_padding_mask):
        # (bs, dim), (bs, dim), (bs, seq_len1, dim), (bs, seq_len)
        entity_latent = entity_latent.unsqueeze(1)  # (bs, 1, dim)
        review_latent = review_latent.unsqueeze(1)  # (bs, 1, dim)
        dialog_latent = dialog_latent[:, -1:, :]  # (bs, 1, dim)

        logits = self._get_logits(entity_latent, review_latent, dialog_latent, conv_review_reps, review_padding_mask) # (bs, 1, nb_tok)
        preds = logits.argmax(dim=-1).long() # (bs, 1)

        return logits, preds
    
    def _get_logits(self, entity_latent, review_latent, dialog_latent, conv_review_reps, review_padding_mask):
        # (bs, seq_len, dim) * 3
        logits = self._get_logits_hs_copy2(entity_latent, review_latent, dialog_latent, conv_review_reps, review_padding_mask) # (bs, seq_len, nb_tok)
        logits = self.weight_logits(logits) # (bs, seq_len, nb_tok)
        return logits # (bs, seq_len, nb_tok)
    
    def weight_logits(self, logits):
        # (bs, seq_len, nb_tok)
        if self.is_weight_logits and self.decoder_token_prob_weight is not None:
            logits = logits * self.decoder_token_prob_weight
            
        return logits

    def _get_logits_hs_copy2(self, entity_latent, review_latent, dialog_latent, conv_review_reps, review_padding_mask):
        # (bs, seq_len, dim) * 3, (bs, nb_review, ffn_size)
        fusion_latent = self.get_fusion_latent2(dialog_latent, conv_review_reps, review_padding_mask) # (bs, seq_len, ffn_size)
        gen_logits = F.linear(fusion_latent, self.decoder_token_embedding.weight) # (bs, seq_len, nb_tok)
        sum_logits = gen_logits # (bs, seq_len, nb_tok)

        return sum_logits # (bs, seq_len, nb_tok)
    
    def get_fusion_latent2(self, dialog_latent, conv_review_reps, review_padding_mask):
        # (bs, seq_len, ffn_size), (bs, nb_review, ffn_size), (bs, nb_review)
        bs, seq_len, _ = dialog_latent.shape
        bs, nb_review, _ = conv_review_reps.shape

        # dialog_latent = dialog_latent.transpose(0, 1).contiguous() # (seq_len, bs, ffn_size)
        # dialog_latent = dialog_latent.unsqueeze(2).expand(-1, -1, nb_review, -1) # (seq_len, bs, nb_review, ffn_size)
        # dialog_latent = dialog_latent.view(-1, self.ffn_size) # (seq_len*bs*nb_review, ffn_size)

        # conv_review_reps = conv_review_reps.view(-1, self.ffn_size) # (bs*nb_review, ffn_size)
        # conv_review_reps = conv_review_reps.expand(seq_len*bs*nb_review, -1) # (seq_len*bs*nb_review, ffn_size)

        # dot_prod = dialog_latent * conv_review_reps # (seq_len*bs*nb_review)
        # dot_prod = dot_prod.view(bs, seq_len, nb_review) # (seq_len, bs, nb_review)

        # weight = F.softmax(atten, dim=-1).type_as(atten) # (seq_len, bs, nb_review)

        dot_prod = dialog_latent.bmm(conv_review_reps.transpose(1, 2)) # (bs, seq_len, nb_review)
        # dot_prod = dot_prod.transpose(0, 1).contiguous() # (seq_len, bs, nb_review)

        attn_mask = review_padding_mask.unsqueeze(1).expand(-1, seq_len, -1) # (bs, seq_len, nb_review)
        dot_prod.masked_fill_(~attn_mask.bool(), neginf(dot_prod.dtype)) # (bs, seq_len, nb_review)
        weight = F.softmax(dot_prod, dim=-1).type_as(conv_review_reps) # (bs, seq_len, nb_review)

        decode_atten_review_reps = weight.bmm(conv_review_reps) # (bs, seq_len, ffn_size)
        decode_atten_review_reps = decode_atten_review_reps.view(bs, seq_len, self.ffn_size) # (bs, seq_len, ffn_size)

        fusion_latent = torch.cat([dialog_latent, decode_atten_review_reps], dim=-1) # (bs, seq_len, ffn_size*2)
        fusion_latent = self.fusion_latent_norm(fusion_latent) # (bs, seq_len, ffn_size)

        return fusion_latent # (bs, seq_len, ffn_size)


class CFSelectionConvModel(nn.Module):
    def __init__(self, opt, device, vocab, side_data, decoder_token_embedding):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        super().__init__()
        self.opt, self.device, self.vocab, self.side_data, self.decoder_token_embedding = opt, device, vocab, side_data, decoder_token_embedding

        # vocab
        self.vocab_size = vocab['vocab_size']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.token_emb_dim = opt['token_emb_dim']
        self.pretrained_embedding = side_data.get('embedding', None)
        # kg
        self.n_word = side_data['word_kg']['n_entity']
        self.kg_name = opt['kg_name']
        self.n_entity = side_data[self.kg_name]['n_entity']
        self.pad_word_idx = vocab['pad_word']
        self.pad_entity_idx = vocab['pad_entity']
        entity_kg = side_data['entity_kg']
        self.n_relation = entity_kg['n_relation']
        entity_edges = entity_kg['edge']
        self.entity_edge_idx, self.entity_edge_type = edge_to_pyg_format(entity_edges, 'RGCN')
        self.entity_edge_idx = self.entity_edge_idx.to(device)
        self.entity_edge_type = self.entity_edge_type.to(device)
        word_edges = side_data['word_kg']['edge']
        self.word_edges = edge_to_pyg_format(word_edges, 'GCN').to(device)
        self.num_bases = opt['num_bases']
        self.kg_emb_dim = opt['kg_emb_dim']
        # transformer
        self.n_heads = opt['n_heads']
        self.n_layers = opt['n_layers']
        self.ffn_size = opt['ffn_size']
        self.dropout = opt['dropout']
        self.attention_dropout = opt['attention_dropout']
        self.relu_dropout = opt['relu_dropout']
        self.learn_positional_embeddings = opt['learn_positional_embeddings']
        self.embeddings_scale = opt['embeddings_scale']
        self.reduction = opt['reduction']
        self.n_positions = opt['n_positions']
        self.response_truncate = opt.get('response_truncate', 20)

        self.build_model()

    def build_model(self):
        self.db_model = DBModel(self.opt, self.device, self.vocab, self.side_data)
        self.review_model = CoarseReviewModelForDecoder(self.opt, self.device, self.vocab, self.side_data)
        self.decoder_model = DecoderCNSelectionModel(self.opt, self.device, self.vocab, self.side_data, self.decoder_token_embedding)

    def model_context(self, conv_encoder, context_tokens):
        encoder_output, encoder_mask = conv_encoder(context_tokens) # (last_hidden_state, mask) = (bs, seq_len, dim), (bs, seq_len)

        return encoder_output, encoder_mask # (last_hidden_state, mask) = (bs, seq_len, dim), (bs, seq_len)
    
    def forward(self, batch, mode, conv_encoder, kgModel, reviewModel):
        # converse
        context_tokens, context_entities, response = \
            batch['context_tokens'], batch['context_entities'], batch['response']
        entity_padding_mask = ~context_entities.eq(self.pad_entity_idx)  # (bs, entity_len)

        encoder_output, encoder_mask = self.model_context(conv_encoder, context_tokens) # (bs, seq_len, dim), (bs, seq_len)
        
        # (bs, dim), (bs, n_context_entities, dim), (bs, ffn_size), (bs, n_context_entities, ffn_size)
        entity_attn_rep, entity_representations, conv_entity_emb, conv_entity_reps  = self.db_model(batch, mode, kgModel)
        
        # (bs, ffn_size), (bs, n_review, ffn_size), (bs, nb_review), (bs, n_review, seq_len3, dim), (bs, n_review, seq_len3)
        conv_review_emb, conv_review_reps, review_padding_mask, review_token_reps, review_token_padding_mask = self.review_model(
            batch, mode, reviewModel)

        loss, preds = self.decoder_model(
            mode, 
            encoder_output, encoder_mask, 
            conv_entity_emb, conv_entity_reps, entity_padding_mask, 
            conv_review_emb, conv_review_reps, review_padding_mask, 
            review_token_reps, review_token_padding_mask,
            response)
        
        return loss, preds
    