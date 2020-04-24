import  collections

from texthub.utils import build_from_cfg
from  ..registry import PIPELINES


class Compose(object):
    """
    类似torchvision.transforms.compose的功能,将所有pipeline串起来
    """
    def __init__(self,transforms):
        assert  isinstance(transforms,collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            #从config中加载
            if isinstance(transform,dict):
                transform = build_from_cfg(transform,PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError("transfrom {} must be callble or a config dict".format(transform))
    def __call__(self,data):
        #data 以dict的形式,方便每个pipeline得到它想要的data
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string