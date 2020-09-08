import torch.nn as nn
from .base import  BaseDetector
from ..registry import DETECTORS
from ..builder import build_backbone,build_neck,build_head

@DETECTORS.register_module
class DBDetector(BaseDetector):

    def __init__(
            self,
            backbone,
            neck=None,
            det_head = None,
            train_cfg = None,
            test_cfg = None,
            pretrained = None):
        super(DBDetector,self).__init__()
        self.backbone = build_backbone(backbone)
        self.neck =  None
        if neck is not None:
            self.neck = build_neck(neck)
        self.det_head = build_head(det_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained)

    def extract_feat(self, data:dict):
        img_features = data.get('img')
        img_features = self.backbone(img_features)
        if self.neck!=None:
            img_features =self.neck(img_features)
        data["img"] = img_features
        return data

    def init_weights(self, pretrained=None):
        if hasattr(self.backbone, "init_weights"):
            self.backbone.init_weights(True)
        if self.neck!=None and hasattr(self.neck, "init_weights"):
            self.neck.init_weights(pretrained)
        self.det_head.init_weights()

    def forward(self,data:dict,return_loss=True, **kwargs):
        """

        """
        if return_loss:
            outputs = self.forward_train(data, **kwargs)
        else:
            outputs =  self.forward_test(data, **kwargs)
        return  outputs


    def forward_train(self,
                      data:dict,
                      **kwargs):
        data = self.extract_feat(data)
        losses = self.det_head(data, return_loss=True)
        ##在runner中loss 回传
        return losses

    def forward_test(self,data:dict,**kwargs):
        data = self.extract_feat(data)
        outs = self.det_head(data, return_loss=False)
        return outs

    def postprocess(self,preds):
        ##预测变为预测框
        return self.det_head.postprocess(preds)

class DBFeatureModel(nn.Module):
    def __init__(self,backbone,neck):
        super(DBFeatureModel, self).__init__()
        self.backbone = backbone
        self.neck = neck