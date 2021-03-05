import torch.nn as nn
from .base import  BaseRecognizer
from ..registry import RECOGNIZERS
from ..builder import build_backbone,build_img_trans,build_head

@RECOGNIZERS.register_module
class SCATTERModel(BaseRecognizer):
    """
    SCATTER: Selective Context Attentional Scene Text Recognizer

    """

    def __init__(
            self,
            backbone,
            transformation = None,
            label_head = None,
            train_cfg = None,
            test_cfg = None,
            pretrained = None,
            ):
        super(SCATTERModel,self).__init__()
        self.backbone = build_backbone(backbone)
        self.img_transformation = None
        if transformation !=None:
            self.img_transformation = build_img_trans(transformation)
        self.label_head = build_head(label_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
       
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        super(SCATTERModel, self).init_weights()
        if self.img_transformation !=None:
            self.img_transformation.init_weights()
        self.backbone.init_weights(pretrained)
        self.label_head.init_weights(pretrained)

    def extract_feat(self, x):
        """Directly extract features from the img backbone
        """
        """ Transformation stage """
        if self.img_transformation:
            x = self.img_transformation(x)

        """ Feature extraction stage """
        visual_feature = self.backbone(x)

        return visual_feature

    def forward_train(self,
                      data:dict,
                      **kwargs):
        img = data.get("img")
        x = self.extract_feat(img)
        data['img'] = x
        losses = self.label_head(data, return_loss=True)
        ##在runner中loss 回传
        return losses

    def forward_test(self,data:dict,**kwargs):
        img = data.get("img")
        x = self.extract_feat(img)
        data['img'] = x
        outs = self.label_head(data,return_loss=False)
        return outs


    def postprocess(self,data):
        return self.label_head.postprocess(data)

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














