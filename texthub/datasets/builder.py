import  copy
from texthub.utils import  build_from_cfg
from .registry import  DATASETS
from texthub.datasets import CharsetDict




def build_dataset(cfg,default_args=None):
    """
    替换charsets
    :param cfg:
    :param default_args:
    :return:
    """
    if "charsets" in cfg.keys():
        charset_type_str = cfg.pop('charsets')
        charset = CharsetDict.get(charset_type_str)
        cfg.setdefault("charsets", charset)
    dataset = build_from_cfg(cfg,DATASETS,default_args)
    return dataset