import  logging
import  six
import  os.path as osp
import  os

from abc import ABCMeta,abstractmethod
import sys
import  torch
from torch.optim.optimizer import Optimizer

from ..utils.checkpoint import load_checkpoint
from .logbuffer import LogBuffer
from .Hooks import (lrupdatehook,BaseHook,get_priority,OptimizerHook,CheckpointHook,IterTimerHook)
from .Hooks.lrupdatehook import LrUpdaterHook
from . import Hooks
from ...utils import get_dist_info


class BaseRunner(metaclass=ABCMeta):
    """The base class of Runner, a training helper for PyTorch.

    All subclasses should implement the following APIs:

    - ``run()``
    - ``train()``
    - ``val()``
    - ``save_checkpoint()``

        Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): It can be either an
            optimizer (in most cases) or a dict of optimizers (in models that
            requires more than one optimizer, e.g., GAN).
        work_dir (str, optional): The working directory to save checkpoints
            and logs. Defaults to None.
        logger (:obj:`logging.Logger`): Logger used during training.
             Defaults to None. (The default value is just for backward
             compatibility)
        meta (dict | None): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
            Defaults to None.
    """
    def __init__(self,model:torch.nn.Module,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger = None,
                 meta = None):
        if batch_processor is not None:
            if not callable(batch_processor):
                raise TypeError('batch_processor must be callable, '
                                'but got {}'.format(type(batch_processor)))

        # check the type of `optimizer`
        if not isinstance(optimizer,dict):
            self.optimizer = self.init_optimizer(optimizer)
        elif not isinstance(optimizer, Optimizer) and optimizer is not None:
            raise TypeError(
                'optimizer must be a torch.optim.Optimizer object or dict or None, but got '.format(type(optimizer)))

        # check the type of `logger`
        if not isinstance(logger, logging.Logger):
            raise TypeError('logger must be a logging.Logger object but got '.format(type(logger)))

        # create work_dir
        if isinstance(work_dir, six.string_types):
            self.work_dir = osp.abspath(work_dir)
            os.makedirs(self.work_dir, exist_ok=True)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')


        self.model = model
        self.batch_processor = batch_processor
        self.logger = logger
        self.meta = meta
        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__
        if meta is not None:
            assert isinstance(meta, dict), '"meta" must be a dict or None'
            self.meta = meta
        else:
            self.meta = None


        ##分布式多进程训练用
        self._rank, self._world_size = get_dist_info()

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

        if logger is None:
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger
        # TODO: Redesign LogBuffer, it is not flexible and elegant enough
        self.log_buffer = LogBuffer()


    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

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

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def val(self):
        pass

    @abstractmethod
    def run(self, data_loaders, workflow, **kwargs):
        pass

    @abstractmethod
    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl,
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        pass

    def current_lr(self):
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

    def current_momentum(self):
        """Get current momentums.

        Returns:
            list[float] | dict[str, list[float]]: Current momentums of all
                param groups. If the runner has a dict of optimizers, this
                method will return a dict.
        """

        def _get_momentum(optimizer):
            momentums = []
            for group in optimizer.param_groups:
                if 'momentum' in group.keys():
                    momentums.append(group['momentum'])
                elif 'betas' in group.keys():
                    momentums.append(group['betas'][0])
                else:
                    momentums.append(0)
            return momentums

        if self.optimizer is None:
            raise RuntimeError(
                'momentum is not applicable because optimizer does not exist.')
        elif isinstance(self.optimizer, torch.optim.Optimizer):
            momentums = _get_momentum(self.optimizer)
        return momentums

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
        else:
            raise TypeError('"args" must be either a Hook object'
                            ' or dict, not {}'.format(type(args)))
    def call_hook(self,fn_name):
        for hook in self._hooks:
            if hasattr(hook,fn_name):
                getattr(hook,fn_name)(self)
    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        if optimizer_config is None:
            optimizer_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}
        self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        self.register_hook(IterTimerHook())
        if log_config is not None:
            self.register_logger_hooks(log_config)

    """
        logger 相关
        """
    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        by_epoch = log_config["by_epoch"]
        for info in log_config['hooks']:
            logger_hook = obj_from_dict(
                info, Hooks, default_args=dict(interval=log_interval,by_epoch=by_epoch))

            self.register_hook(logger_hook, priority='VERY_LOW')

    def init_logger(self, log_dir=None, level=logging.INFO):
        """Init the logger.

        Args:
            log_dir(str, optional): Log file directory. If not specified, no
                log file will be used.
            level (int or str): See the built-in python logging module.

        Returns:
            :obj:`~logging.Logger`: Python logger.
        """
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', level=level)
        logger = logging.getLogger(__name__)
        if log_dir:
            filename = '{}.log'.format(self.timestamp)
            log_file = osp.join(log_dir, filename)
            self._add_file_handler(logger, log_file, level=level)
        return logger

    def _add_file_handler(self,
                          logger,
                          filename=None,
                          mode='w',
                          level=logging.INFO):
        # 往文件中写入logn内容
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return logger

    """
    checkpoints codes
    """
    def load_checkpoint(self,filename,map_location="cpu",strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model,filename,map_location,strict,self.logger)

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

    """
       注册钩子
    """
    def register_lr_hooks(self, lr_config):
        if lr_config==None:
            return
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config)
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            # 在代码中找到这个类
            # hook_name = lr_config['policy'].title() + 'LrUpdaterHook'
            hook_name = lr_config['policy']+ 'LrUpdaterHook'
            lr_config.pop("policy")
            if not hasattr(lrupdatehook, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lrupdatehook, hook_name)
            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError('"lr_config" must be either a LrUpdaterHook object'
                            ' or dict, not {}'.format(type(lr_config)))

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


    
