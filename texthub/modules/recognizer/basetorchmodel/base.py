from abc import ABCMeta,abstractmethod
import torch.nn as nn
import  numpy as np

class BaseTorchRecognizer(nn.Module,metaclass=ABCMeta):
    """Base class for text Recognizer"""
    """
    forward 只用于torch module 中的模型调用,方便转化为onmx或者tensorrt
    """
    def __init__(self):
        super(BaseTorchRecognizer,self).__init__()

    @abstractmethod
    def forward(self,img_tensor):
        pass

    @abstractmethod
    def post_process(self,data):
        pass

