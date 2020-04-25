"""
模块组件,将各个部分功能分开
"""
from .builder import (build_backbone, build_recognizer, build_head,
                      build_neck, build_roi_extractor, build_shared_head)
from .registry import BACKBONES,NECKS,HEADS,RECOGNIZERS

from .recognizer import *
from .backbones import *
from .imgtransformation import *
from .label_heads import *
from .sequencerecognition import *
__all__ = [
    'BACKBONES', 'NECKS' , 'HEADS',
    "RECOGNIZERS", 'build_backbone', 'build_neck', 'build_roi_extractor',
    'build_shared_head', 'build_head', 'build_recognizer',
]
