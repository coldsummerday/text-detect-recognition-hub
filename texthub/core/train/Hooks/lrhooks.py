from __future__ import division
from math import cos, pi
from  torch.optim.optimizer import Optimizer
from .basehook import BaseHook

"""
根据配置在训练过程中改变学习率
"""



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


class MultiStepLrUpdateHook(BaseLrUpdateHook):
    def __init__(self,
                 iters_list:[int],
                 base_lr:float,
                 lr_gamma:float,
                 by_epoch=True,
                 iter_interval = 200,
              ):
        self.base_lr = base_lr
        self.iters_list = iters_list
        self.lr_gamma = lr_gamma
        self.by_epoch = by_epoch
        self.interval = iter_interval



    def before_train_epoch(self, runner):
        if self.by_epoch:
            progress = runner.epoch
            exp = len(self.iters_list)
            for i, s in enumerate(self.iters_list):
                if progress < s:
                    exp = i
                    break
            lr = self.base_lr * self.lr_gamma ** exp
            self._set_lr(runner, lr)




    def before_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner,self.interval):
            return
        progress = runner.iter
        exp = len(self.iters_list)
        for i, s in enumerate(self.iters_list):
            if progress < s:
                exp = i
                break
        lr = self.base_lr * self.lr_gamma ** exp
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
