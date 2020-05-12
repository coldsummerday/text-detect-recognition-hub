from ..utils import  Registry
BACKBONES = Registry('backbone')
IMGTRANSFORMATIONS = Registry("imgtransformation")
SEQUENCERECOGNITIONS = Registry("sequencerecognition")
RECOGNIZERS = Registry("recognizer")


NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
SHARED_HEADS = Registry('shared_head')
HEADS = Registry('head')
LOSSES = Registry('loss')
DETECTORS = Registry('detector')
