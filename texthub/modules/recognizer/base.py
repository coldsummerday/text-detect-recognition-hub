from abc import ABCMeta,abstractmethod
from texthub.utils import print_log
import torch.nn as nn

class BaseRecognizer(nn.Module,metaclass=ABCMeta):
    """Base class for text Recognizer"""
    def __init__(self):
        super(BaseRecognizer,self).__init__()

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')

    @abstractmethod
    def extract_feat(self, img_tensors):
        pass

    @abstractmethod
    def forward_train(self, data):
        pass

    @abstractmethod
    def forward_test(self,data):
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def init_weights(self, pretrained=None):
        pass
