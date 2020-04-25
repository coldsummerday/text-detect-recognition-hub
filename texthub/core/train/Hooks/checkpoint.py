from .basehook import BaseHook


class CheckpointHook(BaseHook):

    def __init__(self,
                 interval=-1,
                 save_optimizer=True,
                 out_dir=None,
                 **kwargs):
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.args = kwargs

    ##用于每个interval 保存checkpoint
    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return

        if not self.out_dir:
            self.out_dir = runner.work_dir
        runner.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)