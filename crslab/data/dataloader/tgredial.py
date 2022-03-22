# @Time   : 2020/12/9
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE:
# @Time   : 2020/12/29, 2020/12/15
# @Author : Xiaolei Wang, Yuanhang Zhou
# @Email  : wxl1999@foxmail.com, sdzyh002@gmail

from crslab.evaluator import conv
import random
from random import shuffle
from copy import deepcopy
from collections import defaultdict
import numpy as np

import os
import json
import torch
from math import ceil
from tqdm import tqdm
import pickle
import ipdb
from typing import List, Dict
from copy import deepcopy
from loguru import logger
from numpy.core.arrayprint import _extendLine

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt


def get_mean(ListInt):
    return np.mean(np.array(ListInt if ListInt else [0]))


class TGReDialDataLoader(BaseDataLoader):
    """Dataloader for model TGReDial.

    Notes:
        You can set the following parameters in config:

        - ``'context_truncate'``: the maximum length of context.
        - ``'response_truncate'``: the maximum length of response.
        - ``'entity_truncate'``: the maximum length of mentioned entities in context.
        - ``'word_truncate'``: the maximum length of mentioned words in context.
        - ``'item_truncate'``: the maximum length of mentioned items in context.

        The following values must be specified in ``vocab``:

        - ``'pad'``
        - ``'start'``
        - ``'end'``
        - ``'unk'``
        - ``'pad_entity'``
        - ``'pad_word'``

        the above values specify the id of needed special token.

        - ``'ind2tok'``: map from index to token.
        - ``'tok2ind'``: map from token to index.
        - ``'vocab_size'``: size of vocab.
        - ``'id2entity'``: map from index to entity.
        - ``'n_entity'``: number of entities in the entity KG of dataset.
        - ``'sent_split'`` (optional): token used to split sentence. Defaults to ``'end'``.
        - ``'word_split'`` (optional): token used to split word. Defaults to ``'end'``.
        - ``'pad_topic'`` (optional): token used to pad topic.
        - ``'ind2topic'`` (optional): map from index to topic.

    """

    def __init__(self, subset, opt, dataset, vocab, side_data):
        """

        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: data for model.
            vocab (dict): all kinds of useful size, idx and map between token and idx.

        """
        super().__init__(opt, dataset, side_data)
        self.subset = subset
        self.side_data = side_data

        self.kg_name = opt['kg_name']
        self.n_entity = side_data[self.kg_name]['n_entity']
        
        self.item_size = self.n_entity
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.unk_token_idx = vocab['unk']
        self.conv_bos_id = vocab['start']
        self.cls_id = vocab['start']
        self.sep_id = vocab['end']
        self.sent_split_idx = vocab.get('sent_split', vocab['end'])
        self.word_split_idx = vocab.get('word_split', vocab['end'])
        self.pad_entity_idx = vocab['pad_entity']
        self.pad_word_idx = vocab['pad_word']
        self.pad_topic_idx = vocab.get('pad_topic', None)

        self.tok2ind = vocab['tok2ind']
        self.ind2tok = vocab['ind2tok']
        self.id2entity = side_data[self.kg_name]['id2entity']
        self.ind2topic = vocab.get('ind2topic', None)

        self.item_ids = side_data['rec']['item_entity_ids'] \
            if 'rec' in side_data else side_data['item_entity_ids']
        
        self.context_truncate = opt.get('context_truncate', 256)
        self.info_truncate = opt.get('info_truncate', 128)
        self.extended_context_truncate = self.context_truncate + self.info_truncate
        
        self.response_truncate = opt.get('response_truncate', None)
        self.entity_truncate = opt.get('entity_truncate', None)
        self.word_truncate = opt.get('word_truncate', None)
        self.item_truncate = opt.get('item_truncate', None)
        self.dataset_name = opt.get('dataset_name', None)
        self.show_input = opt.get('show_input', None)

        if 'rec' in self.opt:
            self.rec_optim_opt = deepcopy(self.opt['rec'])
            self.rec_epoch = self.rec_optim_opt['epoch']
            self.rec_batch_size = self.rec_optim_opt['batch_size']
        
        self.is_sent_split = self.opt['is_sent_split']
        self.nb_review = self.opt['nb_review']

        path = f'data/dataset/{self.opt["dataset"].lower()}/{self.opt["tokenize"]["rec"]}/{self.subset}_conv_idx_to_review_info.pkl'
        self.conv_idx_to_review_info = pickle.load(open(path, 'rb'))
        logger.info(f'load {len(self.conv_idx_to_review_info)} conv_idx_to_review_info')
        self.dataset = self.malloc_information(self.dataset)
    
    def malloc_information(self, dataset):
        result_dataset = []

        for i, conv_dict in enumerate(dataset):
            conv_dict['use_infoListListInt'] = []
            conv_dict['selected_entityIds'] = []

            result_dataset.append(conv_dict)

        return result_dataset

    def rec_process_fn(self, *args, **kwargs):
        augment_dataset = self.rec_process_fn_given_dataset(self.dataset)

        return augment_dataset
    
    def rec_process_fn_given_dataset(self, dataset):
        """如果response里有多个要推荐的电影，则每个电影分别作为推荐任务的ground truth，生成一个数据样本"""
        augment_dataset = []
        
        logger.info('[rec_process_fn_given_dataset]')
        for conv_dict in tqdm(dataset):
            for movie in conv_dict['items']:
                augment_conv_dict = deepcopy(conv_dict)
                augment_conv_dict['item'] = movie
                augment_dataset.append(augment_conv_dict)
        
        return augment_dataset
    
    def _get_original_context_for_rec(self, context_tokens):
        # Insert special token into context. And flat the context.
        # Args:
        #     context_tokens (list of list int): 
        # Returns:
        #     compat_context (list int): 
        
        compact_context = []
        for i, utterance in enumerate(context_tokens):
            utterance = deepcopy(utterance)
            if i != 0 and self.is_sent_split:
                utterance.insert(0, self.sent_split_idx)
            compact_context.append(utterance)
        compat_context = truncate(merge_utt(compact_context),
                                  self.context_truncate - 2,
                                  truncate_tail=False)
        compat_context = add_start_end_token_idx(compat_context,
                                                 self.start_token_idx,
                                                 self.end_token_idx)
        return compat_context  # List[int]

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_size)
        while item in item_set:
            item = random.randint(1, self.item_size)
        return item

    def _process_history(self, context_items, item_id=None):
        input_ids = truncate(context_items,
                             max_length=self.item_truncate,
                             truncate_tail=False)
        input_mask = [1] * len(input_ids)
        sample_negs = []
        seq_set = set(input_ids)
        for _ in input_ids:
            sample_negs.append(self._neg_sample(seq_set))

        if item_id is not None:
            target_pos = input_ids[1:] + [item_id]
            return input_ids, target_pos, input_mask, sample_negs
        else:
            return input_ids, input_mask, sample_negs

    def _convert_sentence_list_from_ind_to_tok(self, sent_list):
        sent_in_tok_list = []
        for sent_in_ind in sent_list:
            sent_in_tok = [self.ind2tok[ind] for ind in sent_in_ind]
            sent_in_tok_list.append(sent_in_tok)
        return sent_in_tok_list
    
    def _show_sentence_list(self, sent_list):
        if isinstance(sent_list[0], list) and isinstance(sent_list[0][0], str):
            for sent in sent_list:
                logger.info('\t{}'.format(sent))
        else:
           sent_list = self._convert_sentence_list_from_ind_to_tok(sent_list)
           self._show_sentence_list(sent_list)

    def show_single_sentence(self, sentence: List[int], context_desc: str):
        # assert all([isinstance(sentence, list), 
        #             isinstance(sentence[0], int)])
        if self.show_input:
            context_in_token_type = [self.ind2tok[ind] for ind in sentence]
            logger.info("[{}] \n\t content = {}".format(context_desc, context_in_token_type))
            logger.info('\tLength = {}'.format(len(sentence)))

    def build_infoListListInt_for_rec(self, conv_dict):
        infoListListInt = []

        selected_entityIds = conv_dict['selected_entityIds']
        infoListListInt = conv_dict['use_infoListListInt']
        
        for i in range(len(infoListListInt)):
            infoListListInt[i] = truncate(infoListListInt[i], self.info_truncate)
        return selected_entityIds, infoListListInt

    def build_item_seq_for_session_rec(self, conv_dict):
        item_id = conv_dict['item']
        interaction_history = conv_dict['context_items']
        if 'interaction_history' in conv_dict:
            interaction_history = conv_dict['interaction_history'] + interaction_history

        input_ids, target_pos, input_mask, sample_negs = self._process_history(
            interaction_history, item_id)

        return input_ids, target_pos, input_mask, sample_negs

    def rec_batchify(self, batch):
        finish_batch = self.rec_batchify_default(batch)
        finish_batch = self.rec_batchify_pipeline_add_disInfo(finish_batch, batch)
        finish_batch = self.rec_batchify_pipeline_add_entities_mask(finish_batch, batch)
        finish_batch = self.rec_batchify_pipeline_add_words(finish_batch, batch)
        finish_batch = self.rec_batchify_pipeline_add_entity(finish_batch, batch)

        return finish_batch

    def rec_batchify_default(self, batch):
        batch_original_context = []
        batch_movie_id = []
        batch_input_ids = []
        batch_target_pos = []
        batch_input_mask = []
        batch_sample_negs = []

        for conv_dict in batch:
            original_context = self._get_original_context_for_rec(conv_dict['context_tokens'])
            batch_original_context.append(original_context)
            batch_movie_id.append(conv_dict['item'])
            input_ids, target_pos, input_mask, sample_negs = self.build_item_seq_for_session_rec(conv_dict)
            batch_input_ids.append(input_ids)
            batch_target_pos.append(target_pos)
            batch_input_mask.append(input_mask)
            batch_sample_negs.append(sample_negs)

        context = padded_tensor(batch_original_context,
                                pad_idx=self.pad_token_idx,
                                pad_tail=True,
                                max_len=self.context_truncate)

        batch = {
            'context': context, 
            'context_mask': (context != 0).long(),
            'context_pad_mask': (context == 0).long(),
            'inter_history_input': padded_tensor(batch_input_ids,
                                                 pad_idx=self.pad_token_idx,
                                                 pad_tail=False,
                                                 max_len=self.item_truncate),
            'inter_history_target': padded_tensor(batch_target_pos,
                                                  pad_idx=self.pad_token_idx,
                                                  pad_tail=False,
                                                  max_len=self.item_truncate),
            'inter_history_mask': padded_tensor(batch_input_mask,
                                                pad_idx=self.pad_token_idx,
                                                pad_tail=False,
                                                max_len=self.item_truncate),
            'inter_history_sample_negs': padded_tensor(batch_sample_negs,
                                                       pad_idx=self.pad_token_idx,
                                                       pad_tail=False,
                                                       max_len=self.item_truncate),
            'movie_to_rec': torch.tensor(batch_movie_id)
        }

        self.show_input_in_rec_batchify(batch)

        return batch

    def rec_batchify_pipeline_add_entity(self, finish_batch, batch):
        batch_context_entities = []

        for conv_dict in batch:
            batch_context_entities.append(conv_dict['context_entities'])

        finish_batch['context_entities'] = batch_context_entities

        return finish_batch
    
    def rec_batchify_pipeline_add_disInfo(self, finish_batch, batch):
        bs = len(batch)

        batch_distInfo = []  # [~bs*nb_review, seq_len]
        batch_NbdistInfo = [] # [bs]
        batch_entities_has_distInfo = [] # [~bs*nb_review]
        batch_er_conv_ids = [] # [~bs*nb_review]

        for i, conv_dict in enumerate(batch):
            selected_entityIds, infoListListInt = self.build_infoListListInt_for_rec(conv_dict)
            batch_distInfo.extend(infoListListInt)
            batch_NbdistInfo.append(len(infoListListInt))
            batch_entities_has_distInfo.extend(selected_entityIds)

            er_conv_ids = [i]*len(infoListListInt)
            batch_er_conv_ids.extend(er_conv_ids)

        if len(batch_distInfo) == 0:
            distInfo = torch.zeros((bs, self.info_truncate), dtype=torch.long)
            batch_NbdistInfo = [1] * bs
            batch_entities_has_distInfo = [-2]
            batch_er_conv_ids = [-2]
        else:
            distInfo = padded_tensor(
                batch_distInfo,
                pad_idx=self.pad_token_idx,
                pad_tail=True,
                max_len=self.info_truncate)

        finish_batch['distInfo'] = distInfo # (~bs*nb_review, seq_len)
        finish_batch['distInfo_mask'] = (distInfo != 0).long() # (~bs*nb_review, seq_len)
        finish_batch['distInfo_pad_mask'] = (distInfo == 0).long() # (~bs*nb_review, seq_len)
        finish_batch['NbdistInfo'] = batch_NbdistInfo # [bs]
        finish_batch['entities_has_distInfo'] = batch_entities_has_distInfo # [~bs*nb_review]
        finish_batch['er_conv_ids'] = batch_er_conv_ids # [~bs*nb_review]

        self.show_input_in_rec_batchify(finish_batch)

        return finish_batch
    
    def rec_batchify_pipeline_add_entities_mask(self, finish_batch, batch):
        batch_entities_mask_in_context = [] # (bs, seq_len)
        batch_entity_masks_in_context = []  # [bs]*(n_eic, seq_len)
        batch_entity_ids_in_context = []  # [~bs*n_eic]
        batch_ect_conv_ids = [] # [~bs*n_eic]

        for i, conv_dict in enumerate(batch):
            entities_mask_in_context = conv_dict['entities_mask_in_context']  # [n_utter, utter_len]
            entities_mask_in_context = self.build_entities_mask_in_context(entities_mask_in_context)  # [seq_len]
            batch_entities_mask_in_context.append(entities_mask_in_context)

            entity_masks_in_context = conv_dict['entity_masks_in_context']  # [n_eic, n_utter, utter_len] or []
            sample_entity_mask_in_context = self.build_sample_entity_mask_in_context(entity_masks_in_context) # (n_eic, seq_len) or ()
            batch_entity_masks_in_context.append(sample_entity_mask_in_context)

            entity_ids_in_context = conv_dict['entity_ids_in_context'] # [n_eic]
            batch_entity_ids_in_context.extend(entity_ids_in_context)

            eic_conv_ids = [i] * len(entity_ids_in_context)
            batch_ect_conv_ids.extend(eic_conv_ids)
            
        entities_mask_in_context = padded_tensor(
            batch_entities_mask_in_context,
            pad_idx=self.pad_token_idx,
            pad_tail=True,
            max_len=self.context_truncate) # (bs, seq_len)

        finish_batch['entities_mask_in_context'] = entities_mask_in_context  # (bs, seq_len)
        finish_batch['entity_masks_in_context'] = batch_entity_masks_in_context  # [bs]*(n_eic, seq_len), n_eic maybe equal to 0
        finish_batch['entity_ids_in_context'] = batch_entity_ids_in_context if batch_entity_ids_in_context else [-1] # [~bs*n_eic]
        finish_batch['eic_conv_ids'] = batch_ect_conv_ids if batch_ect_conv_ids else [-1] # [~bs*n_eic]

        return finish_batch
    
    def rec_batchify_pipeline_add_words(self, finish_batch, batch):
        batch_context_words = []
        for i, conv_dict in enumerate(batch):
            batch_context_words.append(truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))

        context_words = padded_tensor(batch_context_words, self.pad_word_idx, pad_tail=False)
        finish_batch['context_words'] = context_words

        return finish_batch
        
    def convert_ids_to_sentence(self, ids: torch.Tensor):
        return [self.ind2tok.get(int(ind), 'unk') for ind in ids]

    def build_sample_entity_mask_in_context(self, entity_masks_in_context):
        # entity_masks_in_context = [n_eic, n_utter, utter_len]
        if not entity_masks_in_context:
            return torch.tensor([])
             
        sample_entity_mask_in_context = []

        for entity_mask_in_context in entity_masks_in_context:
            # entity_mask_in_context = [n_utter, utter_len]
            entity_mask_in_context = self.build_entities_mask_in_context(entity_mask_in_context) # (seq_len)
            sample_entity_mask_in_context.append(entity_mask_in_context)
        # sample_entity_mask_in_context_len = str([len(entity_mask_in_context) for entity_mask_in_context in sample_entity_mask_in_context])
        # logger.info(f'{sample_entity_mask_in_context_len}')

        sample_entity_mask_in_context = padded_tensor(
            sample_entity_mask_in_context,
            pad_idx=self.pad_token_idx,
            pad_tail=True,
            max_len=self.context_truncate)

        return sample_entity_mask_in_context # (n_eic, seq_len)
    
    def build_entities_mask_in_context(self, entities_mask_in_context: List[List[int]]):
        entities_mask_in_context = self._get_original_context_for_rec(entities_mask_in_context) # (seq_len)
        entities_mask_in_context = torch.LongTensor(entities_mask_in_context)
        entities_mask_in_context = (entities_mask_in_context == -1).long()

        return entities_mask_in_context
    
    def show_input_in_rec_batchify(self, batch):
        if self.opt['show_input']:
            logger.info('[Show Dialog Context] [Show Movie to rec]')
            for context, movie in zip(batch['context'], batch['movie_to_rec']):
                logger.info(context)
                logger.info([self.ind2tok[int(ind)] for ind in context])
                # logger.info(movie, int(self.id2entity[int(movie)]))

                break

            ipdb.set_trace()

    def rec_interact(self, data):
        context = [self._get_original_context_for_rec(data['context_tokens'])]
        if 'interaction_history' in data:
            context_items = data['interaction_history'] + data['context_items']
        else:
            context_items = data['context_items']
        input_ids, input_mask, sample_negs = self._process_history(context_items)
        input_ids, input_mask, sample_negs = [input_ids], [input_mask], [sample_negs]

        context = padded_tensor(context,
                                self.pad_token_idx,
                                max_len=self.context_truncate)
        mask = (context != 0).long()

        return (context, mask,
                padded_tensor(input_ids,
                              pad_idx=self.pad_token_idx,
                              pad_tail=False,
                              max_len=self.item_truncate),
                None,
                padded_tensor(input_mask,
                              pad_idx=self.pad_token_idx,
                              pad_tail=False,
                              max_len=self.item_truncate),
                padded_tensor(sample_negs,
                              pad_idx=self.pad_token_idx,
                              pad_tail=False,
                              max_len=self.item_truncate),
                None)
    
    def build_conv_context_tokens(self, conv_dict):
        context_tokens = [utter + [self.conv_bos_id] for utter in conv_dict['context_tokens']]
        context_tokens[-1] = context_tokens[-1][:-1]
        context_tokens = truncate(merge_utt(context_tokens), max_length=self.context_truncate, truncate_tail=False)

        return context_tokens

    def build_conv_response(self, conv_dict):
        response = add_start_end_token_idx(
            truncate(conv_dict['response'], max_length=self.response_truncate - 2),
            start_token_idx=self.start_token_idx,
            end_token_idx=self.end_token_idx
        )

        return response

    def build_context_entities(self, conv_dict):
        context_entities = truncate(
            conv_dict['context_entities'],
            self.entity_truncate,
            truncate_tail=False)

        return context_entities
  
    def build_context_entities_kbrd(self, conv_dict):
        context_entities_kbrd = conv_dict['context_entities']

        return context_entities_kbrd

    def build_context_words(self, conv_dict):
        context_words = truncate(
            conv_dict['context_words'],
            self.word_truncate,
            truncate_tail=False)

        return context_words

    def build_enhanced_topic(self, conv_dict):
        enhanced_topic = []
        if 'target' in conv_dict:
            for target_policy in conv_dict['target']:
                topic_variable = target_policy[1]
                if isinstance(topic_variable, list):
                    for topic in topic_variable:
                        enhanced_topic.append(topic)
            enhanced_topic = [[
                self.tok2ind.get(token, self.unk_token_idx) for token in self.ind2topic[topic_id]
            ] for topic_id in enhanced_topic]
            enhanced_topic = merge_utt(enhanced_topic, self.word_split_idx, False, self.sent_split_idx)
        
        return enhanced_topic

    def build_enhanced_movie(self, conv_dict):
        enhanced_movie = []
        if 'items' in conv_dict:
            for movie_id in conv_dict['items']:
                enhanced_movie.append(movie_id)
            enhanced_movie = [
                [self.tok2ind.get(token, self.unk_token_idx) for token in self.id2entity[movie_id].split('（')[0]]
                for movie_id in enhanced_movie]
            enhanced_movie = truncate(merge_utt(enhanced_movie, self.word_split_idx, self.sent_split_idx),
                                        self.item_truncate, truncate_tail=False)
        
        return enhanced_movie
    
    def build_enhanced_context_tokens(self, enhanced_movie, enhanced_topic, batch_context_tokens):
        if len(enhanced_movie) != 0:
            enhanced_context_tokens = enhanced_movie + truncate(batch_context_tokens[-1],
                                                                max_length=self.context_truncate - len(
                                                                    enhanced_movie), truncate_tail=False)
        elif len(enhanced_topic) != 0:
            enhanced_context_tokens = enhanced_topic + truncate(batch_context_tokens[-1],
                                                                max_length=self.context_truncate - len(
                                                                    enhanced_topic), truncate_tail=False)
        else:
            enhanced_context_tokens = batch_context_tokens[-1]

        return enhanced_context_tokens

    def conv_batchify_default(self, batch):
        batch_context_tokens = []
        batch_enhanced_context_tokens = []
        batch_response = []
        batch_context_entities = []
        batch_context_entities_kbrd = []
        batch_context_words = []
        for conv_dict in batch:
            context_tokens = self.build_conv_context_tokens(conv_dict)
            batch_context_tokens.append(context_tokens)

            response = self.build_conv_response(conv_dict)
            batch_response.append(response)

            context_entities = self.build_context_entities(conv_dict)
            batch_context_entities.append(context_entities)

            context_words = self.build_context_words(conv_dict)
            batch_context_words.append(context_words)

            # enhanced_topic = self.build_enhanced_topic(conv_dict)
            # enhanced_movie = self.build_enhanced_movie(conv_dict)

            # enhanced_context_tokens = self.build_enhanced_context_tokens(enhanced_movie, enhanced_topic, batch_context_tokens)
            # batch_enhanced_context_tokens.append(enhanced_context_tokens)

            context_entities_kbrd = self.build_context_entities_kbrd(conv_dict)
            batch_context_entities_kbrd.append(context_entities_kbrd)

        batch_context_tokens = padded_tensor(items=batch_context_tokens,
                                             pad_idx=self.pad_token_idx,
                                             max_len=self.context_truncate,
                                             pad_tail=False)
        batch_response = padded_tensor(batch_response,
                                       pad_idx=self.pad_token_idx,
                                       max_len=self.response_truncate,
                                       pad_tail=True)
        batch_input_ids = torch.cat((batch_context_tokens, batch_response), dim=1)
        # batch_enhanced_context_tokens = padded_tensor(items=batch_enhanced_context_tokens,
        #                                               pad_idx=self.pad_token_idx,
        #                                               max_len=self.context_truncate,
        #                                               pad_tail=False)
        # batch_enhanced_input_ids = torch.cat((batch_enhanced_context_tokens, batch_response), dim=1)

        batch = {
            # 'enhanced_input_ids': batch_enhanced_input_ids, 
            # 'enhanced_context_tokens': batch_enhanced_context_tokens,
            'input_ids': batch_input_ids, 
            'context_tokens': batch_context_tokens,
            'context_entities_kbrd': batch_context_entities_kbrd,
            'context_entities': padded_tensor(
                batch_context_entities,
                self.pad_entity_idx,
                pad_tail=False),
            'context_words': padded_tensor(
                batch_context_words,
                self.pad_word_idx,
                pad_tail=False), 
            'response': batch_response
        }
        return batch

    def conv_batchify(self, batch):
        finish_batch = self.conv_batchify_default(batch)
        finish_batch = self.rec_batchify_pipeline_add_disInfo(finish_batch, batch)
        finish_batch = self.rec_batchify_pipeline_add_entities_mask(finish_batch, batch)

        return finish_batch

    def conv_interact(self, data):
        context_tokens = [utter + [self.conv_bos_id] for utter in data['context_tokens']]
        context_tokens[-1] = context_tokens[-1][:-1]
        context_tokens = [truncate(merge_utt(context_tokens), max_length=self.context_truncate, truncate_tail=False)]
        context_tokens = padded_tensor(items=context_tokens,
                                       pad_idx=self.pad_token_idx,
                                       max_len=self.context_truncate,
                                       pad_tail=False)
        context_entities = [truncate(data['context_entities'], self.entity_truncate, truncate_tail=False)]
        context_words = [truncate(data['context_words'], self.word_truncate, truncate_tail=False)]

        return (context_tokens, context_tokens,
                context_tokens, context_tokens,
                padded_tensor(context_entities,
                              self.pad_entity_idx,
                              pad_tail=False),
                padded_tensor(context_words,
                              self.pad_word_idx,
                              pad_tail=False), None)

    def policy_process_fn(self, *args, **kwargs):
        augment_dataset = []
        logger.info('[policy_process_fn]')
        for conv_dict in tqdm(self.dataset):
            for target_policy in conv_dict['target']:
                topic_variable = target_policy[1]
                for topic in topic_variable:
                    augment_conv_dict = deepcopy(conv_dict)
                    augment_conv_dict['target_topic'] = topic
                    augment_dataset.append(augment_conv_dict)
        return augment_dataset

    def policy_batchify(self, batch):
        batch_context = []
        batch_context_policy = []
        batch_user_profile = []
        batch_target = []

        for conv_dict in batch:
            final_topic = conv_dict['final']
            final_topic = [[
                self.tok2ind.get(token, self.unk_token_idx) for token in self.ind2topic[topic_id]
            ] for topic_id in final_topic[1]]
            final_topic = merge_utt(final_topic, self.word_split_idx, False, self.sep_id)

            context = conv_dict['context_tokens']
            context = merge_utt(context,
                                self.sent_split_idx,
                                False,
                                self.sep_id)
            context += final_topic
            context = add_start_end_token_idx(
                truncate(context, max_length=self.context_truncate - 1, truncate_tail=False),
                start_token_idx=self.cls_id)
            batch_context.append(context)

            # [topic, topic, ..., topic]
            context_policy = []
            for policies_one_turn in conv_dict['context_policy']:
                if len(policies_one_turn) != 0:
                    for policy in policies_one_turn:
                        for topic_id in policy[1]:
                            if topic_id != self.pad_topic_idx:
                                policy = []
                                for token in self.ind2topic[topic_id]:
                                    policy.append(self.tok2ind.get(token, self.unk_token_idx))
                                context_policy.append(policy)
            context_policy = merge_utt(context_policy, self.word_split_idx, False)
            context_policy = add_start_end_token_idx(
                context_policy,
                start_token_idx=self.cls_id,
                end_token_idx=self.sep_id)
            context_policy += final_topic
            batch_context_policy.append(context_policy)

            batch_user_profile.extend(conv_dict['user_profile'])

            batch_target.append(conv_dict['target_topic'])

        batch_context = padded_tensor(batch_context,
                                      pad_idx=self.pad_token_idx,
                                      pad_tail=True,
                                      max_len=self.context_truncate)
        batch_cotnext_mask = (batch_context != 0).long()
        batch_context_policy = padded_tensor(batch_context_policy,
                                             pad_idx=self.pad_token_idx,
                                             pad_tail=True)
        batch_context_policy_mask = (batch_context_policy != 0).long()
        batch_user_profile = padded_tensor(batch_user_profile,
                                           pad_idx=self.pad_token_idx,
                                           pad_tail=True)
        batch_user_profile_mask = (batch_user_profile != 0).long()
        batch_target = torch.tensor(batch_target, dtype=torch.long)

        return (batch_context, batch_cotnext_mask, batch_context_policy,
                batch_context_policy_mask, batch_user_profile,
                batch_user_profile_mask, batch_target)

    def _get_original_context_for_rec(self, context_tokens):
        # Insert special token into context. And flat the context.
        # Args:
        #     context_tokens (list of list int): 
        # Returns:
        #     compat_context (list int): 
        
        compact_context = []
        for i, utterance in enumerate(context_tokens):
            utterance = deepcopy(utterance)
            if i != 0 and self.is_sent_split:
                utterance.insert(0, self.sent_split_idx)
            compact_context.append(utterance)
        compat_context = truncate(merge_utt(compact_context),
                                  self.context_truncate - 2,
                                  truncate_tail=False)
        compat_context = add_start_end_token_idx(compat_context,
                                                 self.start_token_idx,
                                                 self.end_token_idx)
        return compat_context  # List[int]

    def add_avi_info_to_init_dataset_u(self):
        extend_dataset = []
        
        logger.info('add_avi_info_to_given_dataset_u')
        for conv_idx, conv_dict in tqdm(enumerate(self.dataset)):
            # List[List[Int]], List[ERId], List[entityId]
            # print(conv_dict.keys())
            selected_infoListListInt = self.conv_idx_to_review_info[str(conv_idx)]['selected_infoListListInt']
            selected_entityIds = self.conv_idx_to_review_info[str(conv_idx)]['selected_entityIds']

            extend_conv_dict = self.add_given_info_to_conv_dict_dr(deepcopy(conv_dict), selected_infoListListInt, selected_entityIds)
            extend_dataset.append(extend_conv_dict)

        return extend_dataset

    def add_given_info_to_conv_dict_dr(self, conv_dict, use_infoListListInt, selected_entityIds):
        conv_dict['use_infoListListInt'] = use_infoListListInt
        conv_dict['selected_entityIds'] = selected_entityIds

        return conv_dict

    def replace_dataset(self, new_dataset):
        logger.info(f'Replace dataset of size {len(self.dataset)} with new_dataset of size {len(new_dataset)}')
        self.dataset = new_dataset
