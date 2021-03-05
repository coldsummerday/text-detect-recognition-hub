import torch
import torch.nn as nn

from ..registry import HEADS
from .labelconverter import CTCLabelConverter
from ..builder import build_loss
@HEADS.register_module
class CTCHead(nn.Module):
    def __init__(self, input_size, charsets,batch_max_length=25,use_baidu_ctc=False,loss=None):
        super(CTCHead, self).__init__()
        self.converter = CTCLabelConverter(charsets)
        self.num_class = len(self.converter.character)
        self.batch_max_length = batch_max_length
        self.use_baidu_ctc = use_baidu_ctc
        if self.use_baidu_ctc:
            # need to install warpctc. see our guideline.
            from warpctc_pytorch import CTCLoss
            self.loss_func = CTCLoss()
        elif loss!=None:
            self.loss_func = build_loss(loss)
        else:
            self.loss_func = torch.nn.CTCLoss(zero_infinity=True)
        self.fc = nn.Linear(input_size,self.num_class)


    def forward(self,data:dict,return_loss:bool,**kwargs):
        if return_loss:
            return self.forward_train(data)
        else:
            return self.forward_test(data)

    def postprocess(self,preds:torch.Tensor):
        batch_size = preds.size(0)
        # Select max probabilty (greedy decoding) then decode index to character
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        _, preds_index = preds.max(2)
        # preds_index = preds_index.view(-1)
        preds_str = self.converter.decode(preds_index, preds_size)
        scores = []
        return preds_str, scores


    def forward_train(self,data:dict):
        img_tensor = data.get("img")
        batch_size = img_tensor.size(0)
        # print(img_tensor.shape)
        device = img_tensor.device
        text = data["label"]
        length = data["length"]
        preds = self.fc(img_tensor)

        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        if self.use_baidu_ctc:
            preds = preds.permute(1, 0, 2)  # to use CTCLoss format
            loss = self.loss_func(preds, text, preds_size, length) / batch_size

        else:
            length = length.long()
            preds_size=  preds_size.long()
            preds = preds.log_softmax(2).permute(1, 0, 2)
            loss = self.loss_func(preds, text, preds_size, length.view(batch_size))
            # print(loss)
            loss = torch.where(torch.isinf(loss), torch.full_like(loss, 6.9), loss)
        return dict(
            loss=loss, ctc_loss=loss
        )


    def forward_test(self,data:dict):
        img_tensor = data.get("img")

        preds = self.fc(img_tensor)
        return preds


class FocalCTCloss(torch.nn.Module):
    def __init__(self,alpha=0.5,gamma=2.0):
        super(FocalCTCloss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.torch_ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
    def forward(self,log_probs, targets, input_lengths, target_lengths):
        loss_ctc = self.torch_ctc_loss(log_probs, targets, input_lengths, target_lengths)
        probability = torch.exp(-loss_ctc)
        focal_ctc_loss = torch.mul(torch.mul(self.alpha,torch.pow((1-probability),self.gamma)),loss_ctc)
        return focal_ctc_loss

