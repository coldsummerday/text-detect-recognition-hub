import torch.nn as  nn
import torch
from ..registry import RECOGNIZERS
from ..builder import build_backbone,build_head,build_neck



@RECOGNIZERS.register_module
class SRNRecognitionModel(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 pretrained = None,**kwargs):
        super(SRNRecognitionModel, self).__init__()
        self.backbone = build_backbone(backbone)
        self.neck = None
        if neck!=None:
            self.neck = build_neck(neck)
        self.head = build_head(head)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(True)
        self.neck.init_weights()
        self.head.init_weights()

    def postprocess(self,data):
        return self.head.postprocess(data)

    def forward(self, data, return_loss=True, **kwargs):
        """
        in the text recognition ,the extra_data is the label
        """
        outputs =None
        if return_loss:
            outputs = self.forward_train(data, **kwargs)

        else:
            outputs =  self.forward_test(data, **kwargs)

        return  outputs

    def forward_train(self,
                      data:dict,
                      **kwargs):
        img = data.get("img")
        x = self.extract_feat(img)
        data['img'] = x
        losses = self.head(data, return_loss=True)
        ##在runner中loss 回传
        return losses

    def forward_test(self,data:dict,**kwargs):
        img = data.get("img")
        x = self.extract_feat(img)
        data['img'] = x
        outs = self.head(data,return_loss=False)
        return outs

    def extract_feat(self, x:torch.Tensor):
        x = self.backbone(x)
        if self.neck:
            x = self.neck(x)
        return x
