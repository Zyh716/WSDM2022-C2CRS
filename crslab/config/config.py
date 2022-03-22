# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2021/1/9
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import json
import os
import time
from pprint import pprint

import yaml
from loguru import logger
from tqdm import tqdm
from os.path import dirname, realpath
from copy import deepcopy

ROOT_PATH = dirname(dirname(dirname(realpath(__file__))))
SAVE_PATH = os.path.join(ROOT_PATH, 'save')
LOG_PATH = os.path.join(ROOT_PATH, 'log')
DATA_PATH = os.path.join(ROOT_PATH, 'data')
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
MODEL_PATH = os.path.join(DATA_PATH, 'model')
PRETRAIN_PATH = os.path.join(MODEL_PATH, 'pretrain')
EMBEDDING_PATH = os.path.join(DATA_PATH, 'embedding')


class Config:
    """Configurator module that load the defined parameters."""

    def __init__(self, args, config_file, gpu='-1', debug=False):
        """Load parameters and set log level.

        Args:
            config_file (str): path to the config file, which should be in ``yaml`` format.
                You can use default config provided in the `Github repo`_, or write it by yourself.
            debug (bool, optional): whether to enable debug function during running. Defaults to False.

        .. _Github repo:
            https://github.com/RUCAIBox/CRSLab

        """

        self.opt = self.load_yaml_configs(config_file)
        # gpu, this line can't run correctlu, so replace this line with 'cuda:gpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        # dataset
        dataset = self.opt['dataset']  # 'TGReDial' or 'ReDial' or others
        self.dataset = dataset
        # if str, represent the each task use same tokenized_dataset. 
        #   elif dict, each key is task name, each value is tokenize type
        tokenize = self.opt['tokenize']  
        if isinstance(tokenize, dict):
            tokenize = ', '.join(tokenize.values())

        # model
        model = self.opt.get('model', None)
        rec_model = self.opt.get('rec_model', None)
        conv_model = self.opt.get('conv_model', None)
        policy_model = self.opt.get('policy_model', None)
        if model:
            model_name = model
        else:
            models = []
            if rec_model:
                models.append(rec_model)
            if conv_model:
                models.append(conv_model)
            if policy_model:
                models.append(policy_model)
            model_name = '_'.join(models)
        self.model_name = model_name
        # self.single_model_name = model_name.split('Multi')[-1].split('Cross')[-1]  # Bert or others
        self.single_model_name = 'BERT' if 'BERT' in self.model_name else self.model_name  # Bert or others
        self.log_prefix = self.single_model_name + '_{}'
        logger.info('[CONFIG] {}, {}'.format(self.model_name, self.single_model_name))


        # update opt
        self.time_stamp = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

        add_dict = {
            'model_name': model_name,
            'single_model_name': self.single_model_name,
            'log_prefix': self.log_prefix if 'Multi' in model_name else self.log_prefix.format(0),
            'dataset_name': dataset,
            'gpu': gpu,
            'time_stamp': self.time_stamp,
        }
        add_dict.update(vars(args))
        self.update_opt(add_dict)

        self.build_path()

        # log
        log_dir = dataset + '_' + model_name + '_' + self.time_stamp  # Expected name='ReDail_MultiBERT_2021-2-13-13-14'
        log_dir = os.path.join(LOG_PATH, log_dir)
        os.makedirs(log_dir, exist_ok=True)
        self.opt['LOG_PATH'] = log_dir

        # log_name = self.opt.get("log_name", dataset + '_' + model_name + '_' + self.time_stamp) + ".log"
        # if not os.path.exists("log"):
        #     os.makedirs("log")
        logger.remove()
        if debug:
            level = 'DEBUG'
        else:
            level = 'INFO'
        
        log_name = self.log_prefix + '.log'
   
        logger.add(
            os.path.join(log_dir, log_name.format('summury')), 
            level=level)
        logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True, level=level)

        logger.info(f"[Dataset: {dataset} tokenized in {tokenize}]")
        if model:
            logger.info(f'[Model: {model}]')
        if rec_model:
            logger.info(f'[Recommendation Model: {rec_model}]')
        if conv_model:
            logger.info(f'[Conversation Model: {conv_model}]')
        if policy_model:
            logger.info(f'[Policy Model: {policy_model}]')
        logger.info("[Config]" + '\n' + json.dumps(self.opt, indent=4))

    def build_path(self):
        save_path = os.path.join(SAVE_PATH, f'{self.dataset}_{self.model_name}'+self.time_stamp)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        self.opt['SAVE_PATH'] = save_path
        logger.info('save_path = {}'.format(save_path))

        self.restore_path = self.opt['restore_path']
        if self.restore_path != '':
            restore_path = os.path.join(SAVE_PATH, self.restore_path)
        else:
            restore_path = os.path.join(SAVE_PATH, f'{self.dataset}_{self.model_name}'+self.time_stamp_for_restore)
        logger.info('restore_path = {}'.format(restore_path))
        if self.opt['restore_system'] and not os.path.exists(restore_path):
            raise ValueError(f'The restore_path {restore_path} is not existing')
        self.opt['RESTORE_PATH'] = restore_path

    def update_opt(self, add_dict):
        for key in add_dict:
            self.opt[key] = add_dict[key]
        
        self.time_stamp_for_restore = self.opt.get('time_stamp_for_restore', 'None')
    
    @staticmethod
    def load_yaml_configs(filename):
        """This function reads ``yaml`` file to build config dictionary

        Args:
            filename (str): path to ``yaml`` config

        Returns:
            dict: config

        """
        config_dict = dict()
        with open(filename, 'r', encoding='utf-8') as f:
            config_dict.update(yaml.safe_load(f.read()))
        return config_dict

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.opt[key] = value

    def __getitem__(self, item):
        if item in self.opt:
            return self.opt[item]
        else:
            return None

    def get(self, item, default=None):
        """Get value of corrsponding item in config

        Args:
            item (str): key to query in config
            default (optional): default value for item if not found in config. Defaults to None.

        Returns:
            value of corrsponding item in config

        """
        if item in self.opt:
            return self.opt[item]
        else:
            return default

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.opt

    def __str__(self):
        return str(self.opt)

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    opt_dict = Config('../../config/crs/c2crs/redial.yaml')
    pprint(opt_dict)
