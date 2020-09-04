from __future__ import division
from math import cos, pi
from  torch.optim.optimizer import Optimizer
from .basehook import BaseHook

"""
根据配置在训练过程中改变学习率
"""

class LrUpdaterHook(BaseHook):
    """LR Scheduler

    Args:
        by_epoch (bool): LR changes epoch by epoch
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
        warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts
    """

    def __init__(self,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 interval=1000,
                 warmup_by_epoch=False):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch
        self.interval = interval

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed

    def _set_lr(self, runner, lr_groups):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(runner.optimizer.param_groups,
                                       lr_groups):
                param_group['lr'] = lr

    def get_lr(self, runner, base_lr):
        raise NotImplementedError

    def get_regular_lr(self, runner):
        if isinstance(runner.optimizer, dict):
            lr_groups = {}
            for k in runner.optimizer.keys():
                _lr_group = [
                    self.get_lr(runner, _base_lr)
                    for _base_lr in self.base_lr[k]
                ]
                lr_groups.update({k: _lr_group})

            return lr_groups
        else:
            return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):
        if self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.regular_lr]
        return warmup_lr



class BaseLrUpdateHook(BaseHook):
    def __init__(self):
        pass

    def _set_lr(self, runner,lr):
        # if isinstance(runner.optimizer, dict):
        #     for k, optim in runner.optimizer.items():
        #         for param_group, lr in zip(optim.param_groups, lr_groups[k]):
        #             param_group['lr'] = lr
        # else:
        #     for param_group, lr in zip(runner.optimizer.param_groups,
        #                                lr_groups):
        #         param_group['lr'] = lr
        assert  isinstance(runner.optimizer,Optimizer)
        for param_group in runner.optimizer.param_groups:
            param_group['lr'] = lr


class WarmupAndDecayLrUpdateHook(BaseLrUpdateHook):
    def __init__(self,
                 base_lr:float,
                 warmup_lr:float,
                 warmup_num:int,
                 lr_gamma:float,
                 by_epoch=True,
                 iter_interval = 200,
                 min_lr = None):
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.warmupnum = warmup_num
        self.lr_gamma = lr_gamma
        self.by_epoch = by_epoch
        self.interval = iter_interval
        self.min_lr =min_lr

    def before_train_epoch(self, runner):
        if self.by_epoch:
            if runner.epoch < self.warmupnum:
                lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * runner.epoch / (self.warmupnum)
            else:
                lr = self.base_lr * (1 - float(runner.epoch) / runner._max_epochs) ** self.lr_gamma
            if self.min_lr:
                lr = max(self.min_lr,lr)
            self._set_lr(runner,lr)

    def before_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner,self.interval):
            return

        if runner.iter < self.warmupnum:
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * runner.iter / (self.warmupnum)
        else:
            lr = self.base_lr * (1 - float(runner.iter) / runner._max_iters) ** self.lr_gamma

        if self.min_lr:
            lr = max(self.min_lr, lr)
        self._set_lr(runner, lr)







# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate
#     # Adapted from PyTorch Imagenet example:
#     # https://github.com/pytorch/examples/blob/master/imagenet/main.py
#     """
#     if epoch < config.warm_up_epoch:
#         lr = 1e-6 + (config.lr - 1e-6) * epoch / (config.warm_up_epoch)
#     else:
#         lr = config.lr * (config.lr_gamma ** (epoch / config.lr_decay_step[0]))
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
#     return lr

# def adjust_learning_rate(optimizer, dataloader, epoch, iter, cfg):
#     schedule = cfg.train_cfg.schedule
#
#     if isinstance(schedule, str):
#         assert schedule == 'polylr', 'Error: schedule should be polylr!'
#         cur_iter = epoch * len(dataloader) + iter
#         max_iter_num = cfg.train_cfg.epoch * len(dataloader)
#         lr = cfg.train_cfg.lr * (1 - float(cur_iter) / max_iter_num) ** 0.9
#     elif isinstance(schedule, tuple):
#         lr = cfg.train_cfg.lr
#         for i in range(len(schedule)):
#             if epoch < schedule[i]:
#                 break
#             lr = lr * 0.1
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
