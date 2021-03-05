import torch
import torch.nn as nn
import numpy as np
import time
import warnings

from ..registry import LOSSES
from ...ops.en_ctc import ENCTCLossFunction

@LOSSES.register_module
class ENCTCcudaLoss(nn.Module):
    def __init__(self, blank=0, h_rate = 0.2,reduction='mean', zero_infinity=True):
        super(ENCTCcudaLoss, self).__init__()
        self.blank = blank
        self.zero_infinity = zero_infinity
        self.reduction = reduction
        self.h_rate = h_rate

    #loss = enctc_loss(input, target, input_lengths, target_lengths,0.2,0,True)
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        loss_value= ENCTCLossFunction.apply(log_probs, targets, input_lengths, target_lengths, self.h_rate,self.blank,
                            self.zero_infinity)
        return loss_value