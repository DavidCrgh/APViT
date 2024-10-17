import argparse
import copy
import os
import os.path as osp
import time
import random

# Suppress MMCV deprecation warnings
import warnings
warnings.filterwarnings(
    action='ignore', 
    category=UserWarning, 
    message='On January 1, 2023, MMCV will release v2.0.0')

# Suppress Albumentations update warnings if running in online env
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') or os.environ.get('COLAB_GPU', ''):
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import numpy as np
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, CheckpointLoader

from mmcls import __version__
from mmcls.apis import set_random_seed, train_model
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.utils import collect_env, get_root_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=25235, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--use_fp16', 
        action='store_true', 
        help='Train using half-precision floating point')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def set_num_classes(cfg, num_classes: int):
    """
    Propagates the num_classes option to the config dictionary.

    Args:
        cfg: Config dictionary
        num_classes: Number of classes to set
    """

    if cfg is None:
        raise ValueError('Config dictionary (cfg) cannot be None.')

    if isinstance(num_classes, str):
        num_classes = int(num_classes)
    elif not isinstance(num_classes, int):
        raise ValueError(f'Parameter num_classes must be an integer, got {type(num_classes)}.')

    cfg.num_classes = num_classes
    cfg.data.train.dataset.num_classes = num_classes
    cfg.data.val.num_classes = num_classes
    cfg.data.test.num_classes = num_classes
    cfg.model.head.num_classes = num_classes


def main():
    args = parse_args()
    print(args)

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
        if 'num_classes' in args.options:
            print(f'Overriding num_classes with {args.options["num_classes"]}')
            set_num_classes(cfg, args.options['num_classes'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # Set FP16 training
    if args.use_fp16:
        cfg['fp16'] = dict(loss_scale='dynamic')

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # Init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    # If resume_from is passed, the metadata will be loaded from the 
    # checkpoint
    if args.resume_from:
        checkpoint = CheckpointLoader.load_checkpoint(
            args.resume_from, 
            map_location='cpu',
            logger=logger)
        
        if checkpoint and 'meta' in checkpoint:
            meta = copy.deepcopy(checkpoint['meta'])
        else:
            warnings.warn('No metadata found in the checkpoint. Creating an empty one.')
            meta = dict()

        del checkpoint
    else:
        meta = dict()

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    model = build_classifier(cfg.model)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 

    # for name, param in model.named_parameters():
    #     print(name)
    # exit()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # val_dataset.pipeline = cfg.data.val.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmcls version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmcls_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    print(f'Train on {len(datasets[0])} samples')
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
    print('Exiting train.py script.')
