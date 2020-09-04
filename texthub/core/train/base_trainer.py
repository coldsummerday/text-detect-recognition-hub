import logging
import six
import os.path as osp
import os
import torch
import sys
from collections import OrderedDict
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from .logbuffer import LogBuffer
from .Hooks import (BaseHook,get_priority,CheckpointHook,IterTimerHook)
from . import Hooks
from ...utils import get_dist_info


class BaseTrainner(object):
    def __init__(self,model:torch.nn.Module,dataloader:torch.utils.data.DataLoader,optimizer:torch.optim=None,
                 work_dir:str= None,
                 log_level=logging.INFO,
                 logger=None,
                 ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer =  self.init_optimizer(optimizer)
        self._word_dir = work_dir

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        if logger is None:
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger

        self.log_buffer = LogBuffer()

        ##分布式多进程训练用
        self._rank, self._world_size = get_dist_info()
        self._hooks =[]

        self.by_epoch = True #true ->max_epochs  ,False->max_iters
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    def run(self,max_number:int,by_epoch=True):
        '''
        by_epoch ->true train number by epoch .->false train number by iter
        '''
        self.by_epoch = by_epoch
        if by_epoch:
            self._max_epochs = max_number
            self._max_iters = max_number * len(self.dataloader)
        else:
            self._max_iters = max_number
            self._max_epochs = (max_number // len(self.dataloader)) +1
        self.logger.info('Start running, work_dir: %s',self.work_dir)

        self._iter = 0

        self.call_hook('before_run')
        self.model.train()
        self.call_hook('before_train_epoch')
        while self._epoch < self._max_epochs:
            for iter,data_batch in enumerate(self.dataloader):
                self._inner_iter = iter

                self.call_hook('before_train_iter')
                ##将数据转成跟model一眼的device
                data_batch = dict_data2model(self.model,data_batch)
                result_loss_dict = self.model(data=data_batch,return_loss=True)
                assert  isinstance(result_loss_dict,dict) and "loss" in result_loss_dict.keys()

                ##收集loss
                loss = result_loss_dict.pop("loss")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                ##保存到_logbuffer中，等待输出
                self.log_buffer.update(parse_loss(result_loss_dict))
                self.log_buffer.update({"loss":loss.mean().item()})

                self.call_hook('after_train_iter')
                self._iter +=1

            self.call_hook('after_train_epoch')
            self._epoch += 1


        self.call_hook("after_run")



    def current_lr(self)->[]:
        """
        Get current learning rates
        :return:
            list:Current learning rate of all params groups
        """
        if self.optimizer is None:
            raise RuntimeError(
                "lr is not applicate because optimizer does not exist."
            )
        return [group['lr'] for group in self.optimizer.param_groups]


    def init_optimizer(self, optimizer):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.

        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD'>
        """
        if isinstance(optimizer, dict):
            optimizer = obj_from_dict(optimizer, torch.optim,
                                      dict(params=self.model.parameters()))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                'but got {}'.format(type(optimizer)))
        return optimizer
    """
    hook相关
    """

    def register_hooks(self,hooks_cfg:[dict]):
        assert isinstance(hooks_cfg,list)
        for hook_cfg in hooks_cfg:
            priority = hook_cfg.pop('priority')
            hook = obj_from_dict(hook_cfg, Hooks)
            self.register_hook(hook,priority=priority)


    def register_hook(self,hook,priority="NORMAL"):
        """
        Register a hook into the hook list
        :param hook: (:obj 'BaseHook')
        :param priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        :return:
        """
        assert  isinstance(hook,BaseHook)
        priority_int = get_priority(priority)
        hook.priority = priority_int
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority_int >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def build_hook(self, args, hook_type=None):
        if isinstance(args, BaseHook):
            return args
        elif isinstance(args, dict):
            assert issubclass(hook_type, BaseHook)
            return hook_type(**args)
    def call_hook(self,fn_name):
        for hook in self._hooks:
            if hasattr(hook,fn_name):
                getattr(hook,fn_name)(self)

    """
        checkpoints codes
        """

    def load_checkpoint(self, filename, map_location="cpu", strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict, self.logger)

    def save_checkpoint(self, out_dir,
                        filename_tmpl="{}_{}_{}.pth",
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        if meta is None:
            meta = dict(epoch=self._epoch + 1, iter=self._iter)
        else:
            meta.update(epoch=self._epoch + 1, iter=self._iter)
        if self.by_epoch:
            filename = filename_tmpl.format(self._model_name, "epoch", self._epoch + 1)
        else:
            filename = filename_tmpl.format(self._model_name, "iter", self._iter)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        if create_symlink:
            symlink(filename, osp.join(out_dir, 'latest.pth'))

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)


    @property
    def work_dir(self):
        return self._word_dir

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters


def dict_data2model(model,data:dict):
    def tocuda(data):
        for key, values in data.items():
            if hasattr(values, 'cuda'):
                data[key] = values.cuda()
        return data
    if hasattr(model, "module"):
        if next(model.module.parameters()).is_cuda:
            data = tocuda(data)
    else:
        if next(model.parameters()).is_cuda:
            data = tocuda(data)

    return data



def parse_loss(loss_dict:dict)->OrderedDict:
    ##收集各项loss,方便输出
    log_vars = OrderedDict()
    for loss_name, loss_value in loss_dict.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean().item()
        # elif isinstance(loss_value, list):
        #     log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor '.format(loss_name))
    return log_vars


def obj_from_dict(info, parent=None, default_args=None):
    """Initialize an object from dict.

    The dict must contain the key "type", which indicates the object type, it
    can be either a string or type, such as "list" or ``list``. Remaining
    fields are treated as the arguments for constructing the object.

    Args:
        info (dict): Object types and arguments.
        parent (:class:`module`): Module which may containing expected object
            classes.
        default_args (dict, optional): Default arguments for initializing the
            object.

    Returns:
        any type: Object built from the dict.
    """
    assert isinstance(info, dict) and 'type' in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type,six.string_types):
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)

def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)