# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/29
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

# UPDATE
# @Time    :   2022/1/1
# @Author  :   Yuanhang Zhou
# @email   :   sdzyh002@gmail.com


from loguru import logger

from .C2CRS_System import C2CRS_System

system_register_table = {
    'C2CRS_Model': C2CRS_System,
}


def get_system(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
               interact=False, debug=False):
    """
    return the system class
    """
    model_name = opt['model_name']
    if model_name in system_register_table:
        logger.info(f'[Building system {model_name}]')
        system = system_register_table[model_name](opt, train_dataloader, valid_dataloader, test_dataloader, vocab,
                                                   side_data, restore_system, interact, debug)
        logger.info(f'[Build system {model_name}]')
        return system
    else:
        raise NotImplementedError('The system with model [{}] in dataset [{}] has not been implemented'.
                                  format(model_name, opt['dataset']))
