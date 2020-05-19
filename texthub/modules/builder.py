from torch import nn
from ..datasets.charsets import CharsetDict
from ..utils import build_from_cfg
from .registry import (BACKBONES, IMGTRANSFORMATIONS,
                       SEQUENCERECOGNITIONS,
                       RECOGNIZERS,NECKS,HEADS,
                       ROI_EXTRACTORS, SHARED_HEADS,DETECTORS)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_img_trans(cfg):
    return build(cfg,IMGTRANSFORMATIONS)

def build_sequence(cfg):
    return build(cfg,SEQUENCERECOGNITIONS)

def build_head(cfg):
    ##TODO:build charsets
    """
        替换charsets
   """
    if "charsets" in cfg.keys():
        charset_type_str = cfg.pop('charsets')
        charset = CharsetDict.get(charset_type_str)
        cfg.setdefault("charsets", charset)
    return build(cfg,HEADS)

def build_recognizer(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, RECOGNIZERS, dict(train_cfg=train_cfg, test_cfg=test_cfg))

def build_detector(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))

def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_roi_extractor(cfg):
    return build(cfg, ROI_EXTRACTORS)


def build_shared_head(cfg):
    return build(cfg, SHARED_HEADS)


# def build_head(cfg):
#     return build(cfg, HEADS)
#
#
# def build_loss(cfg):
#     return build(cfg, LOSSES)
#
#

