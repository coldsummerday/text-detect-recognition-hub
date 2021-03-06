from abc import ABCMeta,abstractmethod
from ...utils import print_log
import torch.nn as nn

class BaseRecognizer(nn.Module,metaclass=ABCMeta):
    """Base class for text Recognizer"""
    def __init__(self):
        super(BaseRecognizer,self).__init__()

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')

    @abstractmethod
    def extract_feat(self, data):
        pass

    @abstractmethod
    def postprocess(self,data):
        pass

    @abstractmethod
    def forward_train(self, data):
        pass

    @abstractmethod
    def forward_test(self,data):
        pass


    @abstractmethod
    def init_weights(self, pretrained=None):
        pass
