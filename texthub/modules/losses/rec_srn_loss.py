

import torch
import torch.nn as nn
from ..registry import LOSSES
@LOSSES.register_module
class RECSRNLoss(nn.Module):
    """
    三个交叉熵loss
    """
    def __init__(self,word_weight=1,vsfd_weight=2.0,gsrm_weight=0.15):
        super(RECSRNLoss, self).__init__()
        self.word_weight = word_weight
        self.vsfd_weight = vsfd_weight
        self.gsrm_weight = gsrm_weight

        self.word_loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.vsfd_loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.gsrm_loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self,predict:torch.Tensor,word_out:torch.Tensor,gsrm_out:torch.Tensor,label_tensor:torch.Tensor)->dict:

        #attention 写法  loss_recog = self.loss_func(probs.view(-1, probs.shape[-1]), target.contiguous().view(-1))
        label = label_tensor.contiguous().view(-1)
        loss_word = self.word_loss_func(word_out.view(-1,word_out.shape[-1]),label)
        loss_gsrm = self.gsrm_loss_func(gsrm_out.view(-1,gsrm_out.shape[-1]),label)
        loss_vsfd = self.vsfd_loss_func(predict.view(-1,predict.shape[-1]),label)
        return dict(
            loss=self.word_weight*loss_word+self.gsrm_weight*loss_gsrm+self.vsfd_weight*loss_vsfd,
            loss_word=loss_word, loss_gsrm=loss_gsrm,loss_vsfd=loss_vsfd
        )
