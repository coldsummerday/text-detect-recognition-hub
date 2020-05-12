from .basehook import BaseHook


class DistSamplerSeedHook(BaseHook):

    def before_epoch(self, runner):
        """
        sampler 中根据epoch  设置随机数种子
        :param runner:
        :return:
        """
        runner.data_loader.sampler.set_epoch(runner.epoch)
