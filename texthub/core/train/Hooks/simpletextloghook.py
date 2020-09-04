
import datetime
from .loghook import  LoggerHook
class SimpleTextLoggerHook(LoggerHook):
    """Base class for logger hooks.

    Args:
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
        reset_flag (bool): Whether to clear the output buffer after logging.
        by_epoch (bool): Whether EpochBasedRunner is used.
    """


    def __init__(self,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag
        self.by_epoch = by_epoch

        #计算剩余时间
        self.time_sec_tot = 0
        self.start_iter = 0


    def before_run(self, runner):
        self.start_iter = runner.iter
        for hook in runner.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break

    def before_epoch(self, runner):
        runner.log_buffer.clear()  # clear logs of last epoch

    def after_train_iter(self, runner):
        if self.by_epoch and self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif not self.by_epoch and self.every_n_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)



        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_train_epoch(self, runner):
        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def log(self,runner):
        log_str_list = []


        if self.by_epoch:
            log_str_list.append("Epoch [{}][{}/{}]".format(runner.epoch+1,runner.inner_iter+1,len(runner.dataloader)))
        else:
            log_str_list.append("Iter [{}/{}]".format(runner.iter+1,runner.max_iters))

        if 'time' in runner.log_buffer.output.keys():
            self.time_sec_tot += (runner.log_buffer.output['time'] * self.interval)
            time_sec_avg = self.time_sec_tot / (
                    runner.iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_str_list.append(f'eta: {eta_str}')

        # only record lr of the first param group
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_str_list.append("lr:{.5f}".format(cur_lr[0]))
        else:
            log_str_list.append("lr:{.5f}".format(cur_lr))

        for name, val in runner.log_buffer.output.items():
            if name in ['time', 'data_time']:
                continue
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_str_list.append("{}:{}".format(name,val))

        runner.logger.info(",".join(log_str_list))




