# @Time    :   2022/1/1
# @Author  :   Yuanhang Zhou
# @email   :   sdzyh002@gmail.com

import os
from math import floor

import torch
from loguru import logger
from typing import List, Dict
from copy import copy, deepcopy
import pickle
import os
import numpy
import ipdb

from crslab.config import PRETRAIN_PATH, SAVE_PATH
from crslab.data import get_dataloader, dataset_language_map
from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt, ind2txt2

import random
from tqdm import tqdm

class C2CRS_System(BaseSystem):
    """This is the system for TGReDial model"""

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.

        """
        super(C2CRS_System, self).__init__(opt, train_dataloader, valid_dataloader,
                                             test_dataloader, vocab, side_data, restore_system, interact, debug)

        self._init_token_attribute(vocab)
        self._init_rec_attribute(side_data, vocab)
        self._init_conv_attribute(side_data, vocab)
        self._init_pretrain_attribute(side_data, vocab)

        self.language = dataset_language_map[self.opt['dataset']]

        self.pertrain_save_epoches = [epoch-1 for epoch in eval(opt['pertrain_save_epoches'])]

    def _init_token_attribute(self, vocab):
        self.ind2tok = vocab['rec']['ind2tok']
        self.end_token_idx = vocab['rec']['end']
        self.unk_token_idx = vocab['rec']['unk']
        self.unk = self.ind2tok.get(self.unk_token_idx, '<unk>')

    def _init_rec_attribute(self, side_data, vocab):
        self.item_ids = side_data['rec']['item_entity_ids']
        self.id2entity = side_data['rec']['entity_kg']['id2entity']
        self.dpath = side_data['rec']['dpath']
        
        self.rec_ind2tok = vocab['rec']['ind2tok']
        self.rec_optim_opt = deepcopy(self.opt['rec'])

        self.rec_batch_size = self.opt['rec_batch_size'] if self.opt['rec_batch_size'] != -1 else self.rec_optim_opt['batch_size']
        self.rec_epoch = self.opt['rec_epoch'] if self.opt['rec_epoch'] != -1 else self.rec_optim_opt['epoch']

    def _init_conv_attribute(self, side_data, vocab):
        self.conv_optim_opt = self.opt['conv']

        if self.conv_optim_opt.get('lr_scheduler', None) and 'Transformers' in self.conv_optim_opt['lr_scheduler']['name']:
            batch_num = 0
            for _ in self.train_dataloader['rec'].get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                batch_num += 1
            conv_training_steps = self.conv_epoch * floor(batch_num / self.conv_optim_opt.get('update_freq', 1))
            self.conv_optim_opt['lr_scheduler']['training_steps'] = conv_training_steps

        self.conv_batch_size = self.opt['conv_batch_size'] if self.opt['conv_batch_size'] != -1 else self.conv_optim_opt['batch_size']
        self.conv_epoch = self.opt['conv_epoch'] if self.opt['conv_epoch'] != -1 else self.conv_optim_opt['epoch']

    def _init_pretrain_attribute(self, side_data, vocab):
        if 'pretrain' in self.opt:
            self.pretrain_optim_opt = deepcopy(self.opt['pretrain'])
            self.pretrain_epoch = self.opt['pretrain_epoch'] if self.opt['pretrain_epoch'] != -1 else self.pretrain_optim_opt['pretrain_epoch']
            self.pretrain_batch_size = self.opt['pretrain_batch_size'] if self.opt['pretrain_batch_size'] != -1 else self.pretrain_optim_opt['batch_size']

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, item in zip(rec_ranks, item_label):
            item = self.item_ids.index(item)
            self.evaluator.rec_evaluate(rec_rank, item)
    
    def rec_evaluate_and_return_score(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        _, fully_rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        fully_rec_ranks = fully_rec_ranks.tolist()
        item_label = item_label.tolist()

        scores = []
        for rec_rank, item in zip(rec_ranks, item_label):
            item = self.item_ids.index(item)
            scores.append(self.evaluator.rec_evaluate_and_return_score(rec_rank, fully_rec_ranks, item, self.opt['score_type']))
        
        return scores, rec_ranks

    def conv_evaluate(self, prediction, response):
        """
        Args:
            prediction: torch.LongTensor, shape=(bs, response_truncate-1)
            response: torch.LongTensor, shape=(bs, response_truncate)

            the first token in response is <|endoftext|>,  it is not in prediction
        """
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str, p_ListStr = ind2txt2(p, self.ind2tok, self.end_token_idx)
            r_str, r_ListStr = ind2txt2(r[1:], self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str], p_ListStr, [r_ListStr])

    def step(self, batch, stage, mode, epoch=-1):
        batch, unbatchify_batch = batch
        self.step_default(batch, stage, mode, epoch)
    
    def step_default(self, batch, stage, mode, epoch=-1):
        """
        stage: ['policy', 'rec', 'conv']
        mode: ['train', 'val', 'test]
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        if stage == 'pretrain_rec':
            loss = self.rec_model.pretrain(batch, mode, epoch)
            if loss:
                if mode == "train":
                    self.backward(loss)
                loss = loss.item()
                self.evaluator.optim_metrics.add("loss", AverageMetric(loss))
        elif stage == 'policy':
            if mode == 'train':
                self.rec_model.train()
            else:
                self.rec_model.eval()

            policy_loss, policy_predict = self.rec_model.guide(batch, mode)
            if mode == "train" and policy_loss is not None:
                self.backward(policy_loss)
            else:
                self.policy_evaluate(policy_predict, batch[-1])
            if isinstance(policy_loss, torch.Tensor):
                policy_loss = policy_loss.item()
                self.evaluator.optim_metrics.add("policy_loss",
                                                 AverageMetric(policy_loss))
        elif stage == 'rec':
            if mode == 'train':
                self.rec_model.train()
            else:
                self.rec_model.eval()

            rec_loss, rec_predict = self.rec_model.recommend(batch, mode)
            if mode == "train":
                self.backward(rec_loss)
            else:
                self.rec_evaluate(rec_predict, batch['movie_to_rec'])
            rec_loss = rec_loss.item()
            self.evaluator.optim_metrics.add("rec_loss",
                                             AverageMetric(rec_loss))
        elif stage == "conv":
            if mode != "test":
                gen_loss, pred = self.rec_model.converse(batch, mode)
                if mode == 'train':
                    self.backward(gen_loss)
                else:
                    self.conv_evaluate(pred, batch['response'])
                gen_loss = gen_loss.item()
                self.evaluator.optim_metrics.add("gen_loss",
                                                 AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                # generate response in rec_model.step
                _, pred = self.rec_model.converse(batch, mode)
                response = batch['response']
                self.conv_evaluate(pred, response)
                self.record_conv_gt_pred(response, pred, epoch)
                self.record_conv_gt(response, pred)
                self.record_conv_pred(response, pred, epoch)
        else:
            raise

    def record_conv_gt_pred(self, batch_response, batch_pred, epoch):
        # (bs, response_truncate), (bs, response_truncate)
        file_writer = self.get_file_writer(f'{epoch}_record_conv_gt_pred', '.txt')

        for response, pred in zip(batch_response, batch_pred):
            response_tok_list = self.convert_tensor_ids_to_tokens(response)
            pred_tok_list = self.convert_tensor_ids_to_tokens(pred)

            file_writer.writelines(' '.join(response_tok_list) + '\n')
            file_writer.writelines(' '.join(pred_tok_list) + '\n')
            file_writer.writelines('\n')

        file_writer.close()

    def record_conv_gt(self, batch_response, batch_pred):
        # (bs, response_truncate), (bs, response_truncate)
        file_writer = self.get_file_writer('record_conv_gt', '.txt')

        for response, pred in zip(batch_response, batch_pred):
            response_tok_list = self.convert_tensor_ids_to_tokens(response)

            file_writer.writelines(' '.join(response_tok_list) + '\n')
            file_writer.writelines('\n')

        file_writer.close()

    def record_conv_pred(self, batch_response, batch_pred, epoch):
        # (bs, response_truncate), (bs, response_truncate)
        file_writer = self.get_file_writer(f'{epoch}_record_conv_pred', '.txt')

        for response, pred in zip(batch_response, batch_pred):
            pred_tok_list = self.convert_tensor_ids_to_tokens(pred)

            file_writer.writelines(' '.join(pred_tok_list) + '\n')
            file_writer.writelines('\n')

        file_writer.close()
    
    def get_file_writer(self, file_keywords: str, file_type: str):
        file_name = file_keywords + file_type
        file_path = os.path.join(self.opt['LOG_PATH'], file_name)
        if os.path.exists(file_path):
            file_writer = open(file_path, 'a', encoding='utf-8')
        else:
            file_writer = open(file_path, 'w', encoding='utf-8')

        return file_writer
    
    def convert_tensor_ids_to_tokens(self, token_ids):
        tokens = []

        token_ids = token_ids.tolist() # List[int]
        if not token_ids:
            return tokens


        for token_id in token_ids:
            if token_id == self.end_token_idx:
                return tokens
            tokens.append(self.ind2tok.get(token_id, self.unk))

        return tokens

    def is_early_stop(self, valid_metric, epoch):
        early_stop_result = self.early_stop(valid_metric)
        # logger.info(f'valid_metric = {valid_metric}, early_stop_result = {early_stop_result}, stop_mode = {self.stop_mode}')
        if early_stop_result == 'Stop':
            return True
        elif early_stop_result == 'New Model':
            self.save_model(epoch=epoch, valid_metric=valid_metric)
        elif early_stop_result == 'Patience':
            pass
            
        return False

    def fit(self):
        self.extend_datasets()
        self.pre_training()
        self.train_recommender_default()
        self.train_conversation_using_rec_model()
    
    def extend_datasets(self):
        extend_train_dataset = self.train_dataloader['rec'].add_avi_info_to_init_dataset_u()
        self.train_dataloader['rec'].replace_dataset(extend_train_dataset)

        extend_train_dataset = self.valid_dataloader['rec'].add_avi_info_to_init_dataset_u()
        self.valid_dataloader['rec'].replace_dataset(extend_train_dataset)

        extend_train_dataset = self.test_dataloader['rec'].add_avi_info_to_init_dataset_u()
        self.test_dataloader['rec'].replace_dataset(extend_train_dataset)

    def pre_training(self):
        self.init_pretrain_optim()
        self.pretrain_recommender_convergence()

    def init_pretrain_optim(self):
        self.pretrain_optim_opt = deepcopy(self.opt['pretrain'])

        # get params and training setting
        bert_param = [p for n, p in self.rec_model.named_parameters() if 'bert' in n]
        other_param = [p for n, p in self.rec_model.named_parameters() if 'bert' not in n]
        params = [{'params': bert_param, 'lr': self.pretrain_optim_opt['lr_bert']},
                  {'params': other_param}]

        logger.info('There are {} bert parameters unit, {} other parameters unit'
            .format(len(bert_param), len(other_param)))

        self.init_optim(deepcopy(self.pretrain_optim_opt), params)
    
    def pretrain_recommender_convergence(self):
        for epoch in range(self.pretrain_epoch):
            self.pretrain_recommender_one_epoch(epoch)

            valid_metric = self.valid_pretrain_recommender(epoch)

            if epoch in self.pertrain_save_epoches:
                self.save_model(post_fix='epoch_{}'.format(epoch), epoch=epoch, valid_metric=valid_metric)

            if self.is_early_stop(valid_metric, epoch):
                break
    
    def pretrain_recommender_one_epoch(self, epoch):
        logger.info(f'[{self.log_prefix}][Recommender | Pretrain | Epoch {str(epoch)}]')
        self.evaluator.reset_metrics()
        for batch in self.train_dataloader['rec'].get_rec_data(self.pretrain_batch_size,
                                                               shuffle=True):
            self.step(batch, stage='pretrain_rec', mode='train', epoch=epoch)
        self.evaluator.report()
    
    def valid_pretrain_recommender(self, epoch):
        logger.info(f'[{self.log_prefix}][Recommender | Valid | Epoch {str(epoch)}]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.valid_dataloader['rec'].get_rec_data(self.pretrain_batch_size, 
                                                                   shuffle=False):
                self.step(batch, stage='pretrain_rec', mode='val', epoch=epoch)
            self.evaluator.report()
            
        metric = self.evaluator.optim_metrics['loss']

        return metric

    def train_recommender_default(self):
        self.init_rec_optim()
        self.train_recommender_convergence()

        # test
        if self.rec_epoch != 0:
            self.restore_model_from_save()
        self.test_recommender('final')

    def init_rec_optim(self):
        self.rec_optim_opt = deepcopy(self.opt['rec'])

        # get params and training setting
        bert_param = [p for n, p in self.rec_model.named_parameters() if 'bert' in n]
        other_param = [p for n, p in self.rec_model.named_parameters() if 'bert' not in n]
        params = [{'params': bert_param, 'lr': self.rec_optim_opt['lr_bert']},
                  {'params': other_param}]

        logger.info('There are {} bert parameters unit, {} other parameters unit'
            .format(len(bert_param), len(other_param)))

        self.init_optim(deepcopy(self.rec_optim_opt), params)

    def train_recommender_convergence(self) -> float:
        for epoch in range(self.rec_epoch):
            self.train_recommender_one_epoch(epoch)

            valid_metric = self.valid_recommender(epoch)

            if self.is_early_stop(valid_metric, epoch):
                break

    def train_recommender_one_epoch(self, epoch):
        logger.info(f'[{self.log_prefix}][Recommender | Train | Epoch {str(epoch)}]')
        self.evaluator.reset_metrics()
        for batch in self.train_dataloader['rec'].get_rec_data(self.rec_batch_size,
                                                               shuffle=True):
            self.step(batch, stage='rec', mode='train', epoch=epoch)
        self.evaluator.report()

    def valid_recommender(self, epoch):
        logger.info(f'[{self.log_prefix}][Recommender | Valid | Epoch {str(epoch)}]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.valid_dataloader['rec'].get_rec_data(self.rec_batch_size, 
                                                                   shuffle=False):
                self.step(batch, stage='rec', mode='val', epoch=epoch)
            self.evaluator.report()
            
        metric = self.evaluator.rec_metrics['hit@1'] + self.evaluator.rec_metrics['hit@50']

        return metric
    
    def test_recommender(self, epoch):
        logger.info(f'[{self.log_prefix}][Recommender | Test ]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader['rec'].get_rec_data(self.rec_batch_size,
                                                                  shuffle=False):
                self.step(batch, stage='rec', mode='test', epoch=epoch)
            self.evaluator.report()

    def train_conversation_using_rec_model(self):
        self.init_optim(deepcopy(self.conv_optim_opt), self.rec_model.parameters())

        if self.opt['freeze_parameters']:
            self.rec_model.freeze_parameters()

        self.train_conversation_convergence()

        if self.conv_epoch != 0:
            self.restore_model_from_save()
        self.test_conversation('final')

    def train_conversation_convergence(self):
        for epoch in range(self.conv_epoch):
            self.train_conversation_one_epoch(epoch)

            valid_metric = self.valid_conversation(epoch)
            self.test_conversation('final')

            if self.is_early_stop(valid_metric, epoch):
                break

    def train_conversation_one_epoch(self, epoch):
        logger.info(f'[{self.log_prefix}][Conversation | Train | epoch {str(epoch)}]')
        self.evaluator.reset_metrics()
        for batch in self.train_dataloader['rec'].get_conv_data(
                batch_size=self.conv_batch_size, shuffle=True):
            self.step(batch, stage='conv', mode='train', epoch=epoch)
        self.evaluator.report()

    def valid_conversation(self, epoch):
        logger.info(f'[{self.log_prefix}][Conversation | Valid | epoch {str(epoch)}]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.valid_dataloader['rec'].get_conv_data(
                    batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='val', epoch=epoch)
            self.evaluator.report()
        valid_metric = self.get_sum_dist_metric()

        # early stop
        return valid_metric

    def get_sum_dist_metric(self):
        sum_dist = 0
        for k in range(1, 5):
            try:
                sum_dist += self.evaluator.gen_metrics[f'dist@{k}']
            except:
                pass

        return sum_dist

    def test_conversation(self, epoch):
        logger.info(f'[{self.log_prefix}][Conversation | Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader['rec'].get_conv_data(
                    batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test', epoch=epoch)
            self.evaluator.report()
    
    def interact(self):
        pass