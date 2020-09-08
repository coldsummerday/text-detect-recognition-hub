import torch.nn as nn
from .base import  BaseDetector
from ..registry import DETECTORS
from ..builder import build_backbone,build_neck,build_head

@DETECTORS.register_module
class PAN(BaseDetector):

    def __init__(
            self,
            backbone,
            neck=None,
            bbox_head = None,
            train_cfg = None,
            test_cfg = None,
            pretrained = None):
        super(PAN,self).__init__()
        backbone = build_backbone(backbone)
        if neck is not None:
            neck = build_neck(neck)
        self.feature_model = PANFeatureModel(backbone,neck)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained)


    def extract_feat(self, data):
        return self.feature_model(data)

    def init_weights(self, pretrained=None):
        self.feature_model.init_weights(pretrained)
        self.bbox_head.init_weights()

    def forward(self,data,return_loss=True, **kwargs):
        """
        in the pan ,the extra_data is  dict {score_maps,training_masks}
        """
        if return_loss:
            outputs = self.forward_train(data, **kwargs)
        else:
            outputs =  self.forward_test(data, **kwargs)
        return  outputs


    def forward_train(self,
                      data:dict,
                      **kwargs):
        img_tensor = data.get('img')
        x = self.extract_feat(img_tensor)
        data['img'] = x
        losses = self.bbox_head(data,return_loss=True)
        ##在runner中loss 回传
        return losses

    def forward_test(self,data:dict,**kwargs):
        img = data.get('img')
        x = self.extract_feat(img)
        data['img']=x
        outs = self.bbox_head(data,return_loss=False)
        return outs

    def postprocess(self,preds):
        ##预测变为预测框
        return self.bbox_head.postprocess(preds)


        
class PANFeatureModel(nn.Module):
    """
    pure torch op model

    easy to change to omnx or tensorrt inferrence
    """
    def __init__(self,backbone,neck):
        super(PANFeatureModel, self).__init__()
        self.backbone = backbone
        self.neck = neck
    def forward(self, x):
        _, _, H, W = x.size()
        x = self.backbone(x)
        if self.neck:
            x = self.neck(x)
        #通过图像上采样获取到跟原图一样的大小的图
        y = nn.functional.interpolate(x,size=(H,W),mode='bilinear', align_corners=True)
        return y

    def init_weights(self,pretrained=None):
        if hasattr(self.backbone,"init_weights"):
            self.backbone.init_weights(True)
        if self.neck and hasattr(self.neck,"init_weights"):
            self.neck.init_weights(pretrained)