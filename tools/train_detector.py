from __future__ import division
import argparse
import copy
import os
import os.path as osp
import time
import sys
this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(osp.join(this_path,'../'))


import torch
from texthub.utils import Config
from texthub.apis import train_detector
from texthub.datasets import  build_dataset
from texthub.modules import build_detector
from texthub.utils import get_root_logger,set_random_seed
from texthub.utils.dist_utils import init_dist



def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume', help='the checkpoint file to resume from')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument("--distributed",default=True,type=bool,help="use DistributedDataParallel to train")
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume is not None:
        cfg.resume_from = args.resume

    cfg.gpus = args.gpus

    if args.distributed:
        """
        pytorch:为单机多卡
        """
        init_dist("pytorch", **cfg.dist_params)



    # create work_dir
    os.makedirs(osp.abspath(cfg.work_dir),exist_ok=True)
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # # log env info
    # env_info_dict = collect_env()
    # env_info = '\n'.join([('{}: {}'.format(k, v))
    #                       for k, v in env_info_dict.items()])
    # dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
    #             dash_line)
    # meta['env_info'] = env_info

    # # log some basic info
    logger.info('Distributed training: {}'.format(args.distributed))
    logger.info('Config:\n{}'.format(cfg.text))

    # set random seeds
    if cfg.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            cfg.seed, args.deterministic))
        set_random_seed(cfg.seed, deterministic=args.deterministic)

    meta['seed'] = cfg.seed

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.val.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            config=cfg.text,
           )

    train_detector(
        model,
        datasets,
        cfg,
        distributed=args.distributed,
        validate=False,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
