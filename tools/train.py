from __future__ import division
import argparse
import copy
import os
import os.path as osp
import time
import sys

this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(osp.join(this_path, '../'))

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel, DataParallel
from texthub.utils import Config
from texthub.datasets import build_dataset
from texthub.modules import build_detector,build_recognizer
from texthub.utils import get_root_logger, set_random_seed
from texthub.utils.dist_utils import init_dist, get_dist_info
from texthub.core.optimizer import build_optimizer
from texthub.core.train import BaseTrainner


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
    parser.add_argument("--distributed", default=1, type=int, help="use DistributedDataParallel to train")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--task",type=str,choices=["reco","detect"],default="detect")

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume is not None:
        cfg.resume_from = args.resume

    cfg.gpus = args.gpus

    if args.distributed==1:
        """
        pytorch:为单机多卡
        """
        init_dist("pytorch", **cfg.dist_params)

    # create work_dir
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()

    # # log some basic info
    logger.info('Distributed training: {}'.format(args.distributed))
    logger.info('Config:\n{}'.format(cfg.text))

    # set random seeds
    if cfg.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            cfg.seed, args.deterministic))
        set_random_seed(cfg.seed, deterministic=args.deterministic)

    meta['seed'] = cfg.seed

    if args.task=="detect":
        model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    elif args.task =="reco":
        model = build_recognizer(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    dataset = build_dataset(cfg.data.train)
    logger = get_root_logger(cfg.log_level)

    if args.distributed:
        rank, world_size = get_dist_info()
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            pin_memory=True,
            drop_last=True,
            sampler=DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        )
        model = DistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True)
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        if torch.cuda.is_available() and cfg.gpus != 0:
            # put model on gpus
            model = DataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build trainner
    optimizer = build_optimizer(model, cfg.optimizer)
    trainer = BaseTrainner(model,data_loader,optimizer,work_dir=cfg.work_dir,logger=logger)

    if cfg.resume_from:
        trainer.resume(cfg.resume_from)

    trainer.register_hooks(cfg.train_hooks)
    trainer.run(max_number=cfg.max_number,by_epoch=cfg.by_epoch)



if __name__ == '__main__':
    main()
