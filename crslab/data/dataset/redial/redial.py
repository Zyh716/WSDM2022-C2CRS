# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2021/1/3, 2020/12/19
# @Author : Kun Zhou, Xiaolei Wang, Yuanhang Zhou
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com, sdzyh002@gmail

# UPDATE
# @Time    :   2022/1/1
# @Author  :   Yuanhang Zhou
# @email   :   sdzyh002@gmail.com

r"""
ReDial
======
References:
    Li, Raymond, et al. `"Towards deep conversational recommendations."`_ in NeurIPS 2018.

.. _`"Towards deep conversational recommendations."`:
   https://papers.nips.cc/paper/2018/hash/800de15c79c8d840f4e78d3af937d4d4-Abstract.html

"""

from crslab.config.config import SAVE_PATH
import json
import torch
import os
import pickle
import numpy as np
from copy import copy
from tqdm import tqdm
from loguru import logger
from collections import defaultdict
import ipdb

from crslab.config import DATASET_PATH
from crslab.data.dataset.base import BaseDataset
from .resources import resources


class ReDialDataset(BaseDataset):
    """

    Attributes:
        train_data: train dataset.
        valid_data: valid dataset.
        test_data: test dataset.
        vocab (dict): ::

            {
                'tok2ind': map from token to index,
                'ind2tok': map from index to token,
                'entity2id': map from entity to index,
                'id2entity': map from index to entity,
                'word2id': map from word to index,
                'vocab_size': len(self.tok2ind),
                'n_entity': max(self.entity2id.values()) + 1,
                'n_word': max(self.word2id.values()) + 1,
            }

    Notes:
        ``'unk'`` must be specified in ``'special_token_idx'`` in ``resources.py``.

    """

    def __init__(self, opt, tokenize, restore=False, save=False):
        """Specify tokenized resource and init base dataset.

        Args:
            opt (Config or dict): config for dataset or the whole system.
            tokenize (str): how to tokenize dataset.
            restore (bool): whether to restore saved dataset which has been processed. Defaults to False.
            save (bool): whether to save dataset after processing. Defaults to False.

        """
        resource = resources[tokenize]
        self.special_token_idx = resource['special_token_idx']
        self.unk_token_idx = self.special_token_idx['unk']

        dpath = os.path.join(DATASET_PATH, "redial", tokenize)
        super().__init__(opt, dpath, resource, restore, save)

    def _load_data(self):
        train_data, valid_data, test_data = self._load_raw_data()
        self._load_vocab()
        self._load_other_data()

        vocab = {
            'tok2ind': self.tok2ind,
            'ind2tok': self.ind2tok,
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'word2id': self.word2id,
            'vocab_size': len(self.tok2ind),
            'n_entity': self.n_entity,
            'n_word': self.n_word,
        }
        vocab.update(self.special_token_idx)

        return train_data, valid_data, test_data, vocab

    def _load_raw_data(self):
        # load train/valid/test data
        with open(os.path.join(self.dpath, 'train_data.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            logger.debug(f"[Load train data from {os.path.join(self.dpath, 'train_data.json')}]")
        with open(os.path.join(self.dpath, 'valid_data.json'), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            logger.debug(f"[Load valid data from {os.path.join(self.dpath, 'valid_data.json')}]")
        with open(os.path.join(self.dpath, 'test_data.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            logger.debug(f"[Load test data from {os.path.join(self.dpath, 'test_data.json')}]")

        return train_data, valid_data, test_data

    def _load_vocab(self):
        self.tok2ind = json.load(open(os.path.join(self.dpath, 'token2id.json'), 'r', encoding='utf-8'))
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}

        logger.debug(f"[Load vocab from {os.path.join(self.dpath, 'token2id.json')}]")
        logger.debug(f"[The size of token2index dictionary is {len(self.tok2ind)}]")
        logger.debug(f"[The size of index2token dictionary is {len(self.ind2tok)}]")

    def _load_other_data(self):
        # dbpedia
        self.entity2id = json.load(
            open(os.path.join(self.dpath, 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        # {head_entity_id: [(relation_id, tail_entity_id)]}
        self.entity_kg = json.load(open(os.path.join(self.dpath, 'dbpedia_subkg.json'), 'r', encoding='utf-8'))
        logger.debug(
            f"[Load entity dictionary and KG from {os.path.join(self.dpath, 'entity2id.json')} and {os.path.join(self.dpath, 'dbpedia_subkg.json')}]")

        # conceptNet
        # {concept: concept_id}
        self.word2id = json.load(open(os.path.join(self.dpath, 'concept2id.json'), 'r', encoding='utf-8'))
        self.n_word = max(self.word2id.values()) + 1
        # {relation\t concept \t concept}
        self.word_kg = open(os.path.join(self.dpath, 'conceptnet_subkg.txt'), 'r', encoding='utf-8')
        logger.debug(
            f"[Load word dictionary and KG from {os.path.join(self.dpath, 'concept2id.json')} and {os.path.join(self.dpath, 'conceptnet_subkg.txt')}]")
        
        filepath = os.path.join(self.dpath, 'redial_context_movie_id2crslab_entityId.json')
        self.redial_context_movie_id2crslab_entityId = json.load(open(filepath))

        self.token_freq_th = self.opt['token_freq_th']
        filepath = os.path.join(self.dpath, 'conv_tokID2freq.json')
        conv_tokID2freq = dict(json.load(open(filepath)))
        self.decoder_token_prob_weight = self.get_decoder_decoder_token_prob_weight(conv_tokID2freq)
    
    def get_decoder_decoder_token_prob_weight(self, conv_tokID2freq):
        decoder_token_prob_weight = []
        nb_reform = 0
        
        for tokID in range(max(list(self.tok2ind.values())) + 1):
            freq = conv_tokID2freq.get(tokID, 1)
            weight = (self.token_freq_th * 1.0) / freq if freq > self.token_freq_th else 1.0
            reform_weight = max(self.opt['coarse_weight_th'], weight)
            if reform_weight != weight:
                nb_reform += 1
            decoder_token_prob_weight.append(reform_weight)

        decoder_token_prob_weight = torch.FloatTensor(decoder_token_prob_weight) # (nb_tok)
        decoder_token_prob_weight = decoder_token_prob_weight.unsqueeze(0).unsqueeze(0) # (1, 1, nb_tok)

        return decoder_token_prob_weight

    def _data_preprocess(self, train_data, valid_data, test_data):
        processed_train_data = self._raw_data_process(train_data)
        logger.debug("[Finish train data process]")
        processed_valid_data = self._raw_data_process(valid_data)
        logger.debug("[Finish valid data process]")
        processed_test_data = self._raw_data_process(test_data)
        logger.debug("[Finish test data process]")
        processed_side_data = self._side_data_process()
        logger.debug("[Finish side data process]")
        return processed_train_data, processed_valid_data, processed_test_data, processed_side_data

    def _raw_data_process(self, raw_data):
        augmented_convs = [self._merge_conv_data(conversation["dialog"]) for conversation in tqdm(raw_data)]
        augmented_conv_dicts = []
        for conv in tqdm(augmented_convs):
            augmented_conv_dicts.extend(self._augment_and_add(conv))
        return augmented_conv_dicts
    
    def _merge_conv_data_default(self, dialog):
        augmented_convs = []
        last_role = None
        for utt in dialog:
            text_token_ids = [self.tok2ind.get(word, self.unk_token_idx) for word in utt["text"]]
            movie_ids = [self.entity2id[movie] for movie in utt['movies'] if movie in self.entity2id]
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if entity in self.entity2id]
            word_ids = [self.word2id[word] for word in utt['word'] if word in self.word2id]

            if utt["role"] == last_role:
                augmented_convs[-1]["text"] += text_token_ids
                augmented_convs[-1]["movie"] += movie_ids
                augmented_convs[-1]["entity"] += entity_ids
                augmented_convs[-1]["word"] += word_ids
            else:
                augmented_convs.append({
                    "role": utt["role"],
                    "text": text_token_ids,
                    "entity": entity_ids,
                    "movie": movie_ids,
                    "word": word_ids
                })
            last_role = utt["role"]

        return augmented_convs

    def _merge_conv_data(self, dialog):
        return self._merge_conv_data_add_entities_mask(dialog)

    def _merge_conv_data_add_entities_mask(self, dialog):
        augmented_convs = []
        last_role = None
        for utt in dialog:
            text_token_ids = [self.tok2ind.get(word, self.unk_token_idx) for word in utt["text"]]
            movie_ids = [self.entity2id[movie] for movie in utt['movies'] if movie in self.entity2id]
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if entity in self.entity2id]
            word_ids = [self.word2id[word] for word in utt['word'] if word in self.word2id]
            entity_ids_in_context, entities_mask_in_context, entity_masks_in_context = self.get_entities_info_in_context_text(utt, text_token_ids)

            if utt["role"] == last_role:
                augmented_convs[-1]["text"] += text_token_ids # [utter_len]
                augmented_convs[-1]["movie"] += movie_ids
                augmented_convs[-1]["entity"] += entity_ids
                augmented_convs[-1]["word"] += word_ids
                augmented_convs[-1]["entities_mask_in_context"] += entities_mask_in_context # [utter_len]
                augmented_convs[-1]["entity_masks_in_context"] += entity_masks_in_context # [n_entities_in_utter_text, utter_len]
                augmented_convs[-1]["entity_ids_in_context"] += entity_ids_in_context # [n_entities_in_utter_text]
            else:
                augmented_convs.append({
                    "role": utt["role"],
                    "text": text_token_ids, # [utter_len]
                    "entity": entity_ids,
                    "movie": movie_ids,
                    "word": word_ids,
                    'entities_mask_in_context': entities_mask_in_context, # [utter_len]
                    'entity_masks_in_context': entity_masks_in_context, # [n_entities_in_utter_text, utter_len]
                    'entity_ids_in_context': entity_ids_in_context, # [n_entities_in_utter_text]
                })
            last_role = utt["role"]

        return augmented_convs
    
    def get_entities_info_in_context_text(self, utt, text_token_ids):
        entity_ids_in_context = []   # [n_entities_in_context_text]
        entities_mask_in_context = []  # [utter_len]
        entity_mask_in_context = []
        entity_masks_in_context = [] # [n_entities_in_context_text, <=utter_len]
        for word in utt["text"]:
            entityId = self.word_is_entity(word)
            if entityId:
                entity_ids_in_context.append(entityId)
                entities_mask_in_context.append(-1)
                entity_mask_in_context.append(-1)
                entity_masks_in_context.append(copy(entity_mask_in_context))
            else:
                entities_mask_in_context.append(0)
                entity_mask_in_context.append(0)
            entity_mask_in_context[-1] = 0
        
        # padding entity_masks_in_context
        utter_len = len(text_token_ids)
        for entity_mask_in_context in entity_masks_in_context:
            entity_mask_in_context = entity_mask_in_context + [0]*(utter_len - len(entity_mask_in_context))
        
        return entity_ids_in_context, entities_mask_in_context, entity_masks_in_context
    
    def word_is_entity(self, word):
        if '@' in word and word[1:].isdigit():
            ID = word[1:]
            if ID in self.redial_context_movie_id2crslab_entityId:
                return self.redial_context_movie_id2crslab_entityId[ID]
        return False

    def _augment_and_add(self, raw_conv_dict):
        return self._augment_and_add_add_entities_mask(raw_conv_dict)

    def _augment_and_add_add_entities_mask(self, raw_conv_dict):
        augmented_conv_dicts = []
        context_tokens, context_entities, context_words, context_items, entities_mask_in_contexts, entity_masks_in_contexts, entity_ids_in_contexts = [], [], [], [], [], [], []
        pad_utters = []
        entity_set, word_set = set(), set()
        for i, conv in enumerate(raw_conv_dict):
            text_tokens, entities, movies, words, entities_mask_in_context, entity_masks_in_context, entity_ids_in_context = \
                conv["text"], conv["entity"], conv["movie"], conv["word"], conv["entities_mask_in_context"], conv['entity_masks_in_context'], conv['entity_ids_in_context']
            if len(context_tokens) > 0:
                conv_dict = {
                    "role": conv['role'],
                    "context_tokens": copy(context_tokens),
                    "response": text_tokens,
                    "context_entities": copy(context_entities),
                    "context_words": copy(context_words),
                    "context_items": copy(context_items),
                    "entities_mask_in_context": copy(entities_mask_in_contexts),
                    "entity_masks_in_context": copy(entity_masks_in_contexts),
                    "entity_ids_in_context": copy(entity_ids_in_contexts),
                    "items": movies,
                }
                augmented_conv_dicts.append(conv_dict)

            context_tokens.append(text_tokens)  # [n_utter, utter_len]
            entities_mask_in_contexts.append(entities_mask_in_context)  # [n_utter, utter_len]
            # entity_masks_in_context = [n_entities_in_utter_text, utter_len]
            padded_entity_masks_in_context = self.padd_entity_masks_in_context(pad_utters, entity_masks_in_context) # [n_entities_in_utter_text, n_utter, utter_len]
            entity_masks_in_contexts.extend(padded_entity_masks_in_context) # [n_entities_in_context_text, n_utter, utter_len]
            entity_ids_in_contexts.extend(entity_ids_in_context)  # [n_entities_in_context_text]
            context_items += movies
            for entity in entities + movies:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)
            for word in words:
                if word not in word_set:
                    word_set.add(word)
                    context_words.append(word)
            pad_utters.append([0]*len(text_tokens))

        return augmented_conv_dicts
    
    def padd_entity_masks_in_context(self, pad_utters, entity_masks_in_context):
        # pad_utters = [n_utter, utter_len]
        # entity_masks_in_context = [n_entities_in_utter_text, utter_len]
        padded_entity_masks_in_context = []
        for entity_mask_in_context in entity_masks_in_context:
            # entity_mask_in_context = [utter_len]
            entity_mask_in_context = pad_utters + [entity_mask_in_context] # [n_utter, utter_len]
            padded_entity_masks_in_context.append(entity_mask_in_context)

        return padded_entity_masks_in_context # [n_entities_in_utter_text, n_utter, utter_len]
    
    def get_pad_utters(self, context_tokens):
        # context_tokens = [n_utter, utter_len]
        pad_utters = []
        for utter in context_tokens:
            pad_utter = [0] * len(utter)
            pad_utters.append(pad_utter)

        return pad_utters

    def _augment_and_add_default(self, raw_conv_dict):
        augmented_conv_dicts = []
        context_tokens, context_entities, context_words, context_items = [], [], [], []
        entity_set, word_set = set(), set()
        for i, conv in enumerate(raw_conv_dict):
            text_tokens, entities, movies, words = conv["text"], conv["entity"], conv["movie"], conv["word"]
            if len(context_tokens) > 0:
                conv_dict = {
                    "role": conv['role'],
                    "context_tokens": copy(context_tokens),
                    "response": text_tokens,
                    "context_entities": copy(context_entities),
                    "context_words": copy(context_words),
                    "context_items": copy(context_items),
                    "items": movies,
                }
                augmented_conv_dicts.append(conv_dict)

            context_tokens.append(text_tokens)
            context_items += movies
            for entity in entities + movies:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)
            for word in words:
                if word not in word_set:
                    word_set.add(word)
                    context_words.append(word)

        return augmented_conv_dicts

    def _side_data_process(self):
        processed_entity_kg = self._entity_kg_process()
        logger.debug("[Finish entity KG process]")

        processed_word_kg = self._word_kg_process()
        logger.debug("[Finish word KG process]")
        
        movie_entity_ids = json.load(open(os.path.join(self.dpath, 'movie_ids.json'), 'r', encoding='utf-8'))
        logger.debug('[Load movie entity ids]')

        side_data = {
            "entity_kg": processed_entity_kg,
            "word_kg": processed_word_kg,
            "item_entity_ids": movie_entity_ids,
            'decoder_token_prob_weight': self.decoder_token_prob_weight
        }
        
        return side_data

    def _entity_kg_process(self, SELF_LOOP_ID=185):
        import re
        def filte_entity(entity):
            entity = entity.split('/')[-1]
            word_list = re.split('_|\(|\)', entity)
            word_list = [word.strip('>') for word in word_list if word != '']

            return [word.lower() for word in word_list if word != '']

        edge_list = []  # [(entity, entity, relation)]
        entity2neighbor = defaultdict(list)  # {entityId: List[entity]}
        for entity in range(self.n_entity):
            if str(entity) not in self.entity_kg:
                continue
            edge_list.append((entity, entity, SELF_LOOP_ID))  # add self loop
            for tail_and_relation in self.entity_kg[str(entity)]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != SELF_LOOP_ID:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

        relation_cnt, relation2id, edges, entities = defaultdict(int), dict(), set(), set()
        for h, t, r in edge_list:
            relation_cnt[r] += 1
        for h, t, r in edge_list:
            if relation_cnt[r] > 1000:
                if r not in relation2id:
                    relation2id[r] = len(relation2id)
                edges.add((h, t, relation2id[r]))
                entities.add(self.id2entity[h])
                entities.add(self.id2entity[t])
            entity2neighbor[h].append(t) 

        return {
            'n_entity': self.n_entity,
            'edge': list(edges),
            'n_relation': len(relation2id),
            'entity': list(entities),
            'entity2neighbor': dict(entity2neighbor),
            'id2entity': {idx: entity for idx, entity in self.id2entity.items() if entity!='None'}
        }

    def _word_kg_process(self):
        edges = set()  # {(entity, entity)}
        entities = set()
        for line in self.word_kg:
            kg = line.strip().split('\t')
            entities.add(kg[1].split('/')[0])
            entities.add(kg[2].split('/')[0])
            e0 = self.word2id[kg[1].split('/')[0]]
            e1 = self.word2id[kg[2].split('/')[0]]
            edges.add((e0, e1))
            edges.add((e1, e0))
        # edge_set = [[co[0] for co in list(edges)], [co[1] for co in list(edges)]]
        return {
            'edge': list(edges),
            'entity': list(entities),
            'n_entity': self.n_word
        }
