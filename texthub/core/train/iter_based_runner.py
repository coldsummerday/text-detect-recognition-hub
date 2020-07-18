import os
import os.path as osp
import time


from .base_runner import BaseRunner
from ..utils.checkpoint import  save_checkpoint
from .Hooks import (OptimizerHook,CheckpointHook,IterTimerHook)
from ...utils import get_dist_info


class IterLoader(object):
    def __init__(self,dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch


    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch +=1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)
        return data

    def __len__(self):
        return len(self._dataloader)

class IterBasedRunner(BaseRunner):
    """Iteration-based Runner.

       This runner train models iteration by iteration.
    """

    def train(self,data_loader,**kwargs):
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self._epoch =  data_loader.epoch
        self.call_hook('before_train_iter')
        data_batch = next(data_loader)
        outputs = self.batch_processor(
            self.model, data_batch, train_mode=True, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('batch_processor() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'],
                                   outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1

    def val(self,data_loader,**kwargs):

        self.model.eval()
        self.mode = "val"
        self.data_loader = data_loader
        self._epoch =  data_loader.epoch
        self.call_hook('before_val_iter')
        data_batch = next(data_loader)
        outputs = self.batch_processor(
            self.model, data_batch, train_mode=True, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('batch_processor() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'],
                                   outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_val_iter')
        self._inner_iter += 1

    def run(self, data_loaders, workflow,max_iters,**kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
            max_iters (int): Total training iterations.
        """
        assert isinstance(data_loaders, list)
        assert len(data_loaders) == len(workflow)
        self._max_iters = max_iters
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_dist_info(), work_dir)
        self.logger.info('workflow: %s, max: %d iters', workflow, max_iters)
        self.call_hook('before_run')

        iter_loaders = [IterLoader(x) for x in data_loaders]
        self.call_hook('before_epoch')

        while self.iter < max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                            format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='iter_{}.pth',
                        meta=None,
                        save_optimizer=True,
                        create_symlink=True):
        """Save checkpoint to file.

        Args:
            out_dir (str): Directory to save checkpoint files.
            filename_tmpl (str, optional): Checkpoint file template.
                Defaults to 'iter_{}.pth'.
            meta (dict, optional): Metadata to be saved in checkpoint.
                Defaults to None.
            save_optimizer (bool, optional): Whether save optimizer.
                Defaults to True.
            create_symlink (bool, optional): Whether create symlink to the
                latest checkpoint file. Defaults to True.
        """
        if meta is None:
            meta = dict(iter=self.iter + 1, epoch=self.epoch + 1)
        elif isinstance(meta, dict):
            meta.update(iter=self.iter + 1, epoch=self.epoch + 1)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if isinstance(self.meta,dict):
            meta.update(self.meta)

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            symlink(filename,osp.join(out_dir,'latest.pth'))
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
        checkpoint_config['by_epoch']= False
        lr_config['by_epoch']= False

        self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        self.register_hook(IterTimerHook())
        if log_config is not None:
            log_config['by_epoch']=False
            self.register_logger_hooks(log_config)




def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)

