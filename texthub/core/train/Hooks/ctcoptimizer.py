# Copyright (c) Open-MMLab. All rights reserved.
import torch
from .basehook import BaseHook

"""
            # (ctc_a) For PyTorch 1.2.0 and 1.3.0. To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
            # https://github.com/jpuigcerver/PyLaia/issues/16
            torch.backends.cudnn.enabled = False
            cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
            torch.backends.cudnn.enabled = True                                                                                                 
"""

class RNNClipGradHook(BaseHook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

    def after_backward(self, runner):
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())

