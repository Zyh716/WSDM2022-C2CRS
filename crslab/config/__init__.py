# -*- encoding: utf-8 -*-
# @Time    :   2020/12/22
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/29
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

"""Config module which loads parameters for the whole system.

Attributes:
    SAVE_PATH (str): where system to save.
    DATASET_PATH (str): where dataset to save.
    MODEL_PATH (str): where model related data to save.
    PRETRAIN_PATH (str): where pretrained model to save.
    EMBEDDING_PATH (str): where pretrained embedding to save, used for evaluate embedding related metrics.
"""

import os
from os.path import dirname, realpath

from .config import Config
from .config import ROOT_PATH
from .config import SAVE_PATH
from .config import DATA_PATH
from .config import DATASET_PATH
from .config import MODEL_PATH
from .config import PRETRAIN_PATH
from .config import EMBEDDING_PATH
