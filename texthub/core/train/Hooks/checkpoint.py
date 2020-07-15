from .basehook import BaseHook
from ....utils.dist_utils import master_only


class CheckpointHook(BaseHook):
    def __init__(self,
                 interval=-1,
                 epoch_mode=True,
                 save_optimizer=True,
                 out_dir=None,
                 **kwargs):
        """
        :param interval:
        :param save_mode: true -> save checkpoint as epoch,false -> as iters
        :param save_optimizer:
        :param out_dir:
        :param kwargs:
        """
        self.save_mode = epoch_mode
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.args = kwargs

    @master_only
    def after_train_iter(self, runner):
        if not self.every_n_iters(runner,self.interval):
            return
        if not self.out_dir:
            self.out_dir = runner.work_dir

        if not self.save_mode:
            runner.save_checkpoint(
                self.out_dir, save_optimizer=self.save_optimizer, **self.args)



    ##用于每个interval 保存checkpoint
    @master_only
    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return

        if not self.out_dir:
            self.out_dir = runner.work_dir
        if self.save_mode:
            runner.save_checkpoint(
                self.out_dir, save_optimizer=self.save_optimizer, **self.args)