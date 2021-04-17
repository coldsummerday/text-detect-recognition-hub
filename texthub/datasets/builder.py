import  copy
from ..utils import  build_from_cfg
from .registry import  DATASETS
from ..datasets import CharsetDict




def build_dataset(cfg,default_args=None):
    """
    替换charsets
    :param cfg:
    :param default_args:
    :return:
    """
    if "charsets" in cfg.keys():
        ##一个arges出现多个charsets的时候，第一次替换掉后形成字符串
        charset_type_str = cfg.pop('charsets')
        if charset_type_str in CharsetDict.keys():
            charset = CharsetDict.get(charset_type_str)
        else:
            charset = charset_type_str
        cfg.setdefault("charsets", charset)
    dataset = build_from_cfg(cfg,DATASETS,default_args)
    return dataset