# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE
# @Time    :   2022/1/1
# @Author  :   Yuanhang Zhou
# @email   :   sdzyh002@gmail.com

import argparse
import warnings

from crslab.config import Config

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='config/recommendation/cobert/tgredial.yaml', help='config file(yaml) path')
    parser.add_argument('-g', '--gpu', type=str, default='1',
                        help='specify gpu id(s) to use, only support using a single gpu now. Defaults to cpu(-1).')
    parser.add_argument('-sd', '--save_data', action='store_true',
                        help='save processed dataset')
    parser.add_argument('-rd', '--restore_data', action='store_true',
                        help='restore processed dataset')
    parser.add_argument('-ss', '--save_system', action='store_true',
                        help='save trained system')
    parser.add_argument('-rs', '--restore_system', action='store_true',
                        help='restore trained system')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='use valid dataset to debug your system')
    parser.add_argument('-i', '--interact', action='store_true',
                        help='interact with your system instead of training')
    parser.add_argument('-ct', '--context_truncate', type=int, default=256)
    parser.add_argument('-it', '--info_truncate', type=int, default=128)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--time_stamp_for_restore', type=str, default='None',
                        help='if restore system, use this time stamp. format likes "2021-07-01-22-05-05"')
    parser.add_argument('--show_input', action='store_true',
                        help='show the input of model in token type')
    parser.add_argument('--model_idx', type=int, default=0,
                        help='如果只有一个模型，这个模型的idx手动赋值')
    parser.add_argument('--score_type', type=str, default='hit',
                        help='hit rank ndcg hit50')
    parser.add_argument('--kg_name', type=str, default='entity_kg',
        choices=['entity_kg', 'word_kg'])
    parser.add_argument('--pretrain_epoch', type=int, default=40)
    parser.add_argument('--rec_epoch', type=int, default=-1)
    parser.add_argument('--conv_epoch', type=int, default=0)
    parser.add_argument('-rbs', '--rec_batch_size', type=int, default=-1)
    parser.add_argument('-pbs', '--pretrain_batch_size', type=int, default=-1)
    parser.add_argument('-cbs', '--conv_batch_size', type=int, default=-1)
    parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
    parser.add_argument('--pertrain_save_epoches', default='[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]', type=str)
    parser.add_argument('--is_sent_split', type=bool, default=True)
    parser.add_argument('--restore_model_postfix', type=str, default='')
    parser.add_argument('--model_file_for_restore', type=str, default='')
    parser.add_argument('--restore_path', type=str, default='')
    parser.add_argument('--coarse_loss_lambda', type=float, default=1.0)
    parser.add_argument('--fine_loss_lambda', type=float, default=1.0)
    parser.add_argument('--coarse_pretrain_epoch', type=float, default=12)
    parser.add_argument('--freeze_parameters', action='store_true')
    parser.add_argument('--freeze_parameters_name', type=str, default='k', help='k | k_r')
    parser.add_argument('--logit_type', type=str, default='default')
    parser.add_argument('--token_freq_th', type=int, default=100)
    parser.add_argument('--is_weight_logits', action='store_true')
    parser.add_argument('--is_coarse_weight_loss', action='store_true')
    parser.add_argument('--coarse_weight_th', type=float, default=0.1)
    parser.add_argument('--nb_review', type=int, default=1)

    args, _ = parser.parse_known_args()
    config = Config(args, args.config, args.gpu, args.debug)

    from crslab.quick_start import run_crslab

    run_crslab(config, args.save_data, args.restore_data, args.save_system, args.restore_system, args.interact,
               args.debug)
