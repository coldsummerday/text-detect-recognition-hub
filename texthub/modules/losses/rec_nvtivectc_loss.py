import torch
import torch.nn as nn
import numpy as np
import time
import warnings

from ..registry import LOSSES
from ...ops.nativectc import NativeCTCLossFunction

@LOSSES.register_module
class NativeCTCcudaLoss(nn.Module):
    def __init__(self, blank=0,reduction='mean', zero_infinity=True):
        super(NativeCTCcudaLoss, self).__init__()
        self.blank = blank
        self.zero_infinity = zero_infinity
        self.reduction = reduction

    #loss = enctc_loss(input, target, input_lengths, target_lengths,0.2,0,True)
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        loss_value= NativeCTCLossFunction.apply(log_probs, targets, input_lengths, target_lengths,self.blank,
                            self.zero_infinity)
        return loss_value