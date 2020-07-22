import torch.nn as nn
from .base import  BaseRecognizer
from ..registry import RECOGNIZERS
from ..builder import build_backbone,build_img_trans,build_sequence,build_head



@RECOGNIZERS.register_module
class FourStageRecoModel(BaseRecognizer):
    """
    four stage recognition model with:trans->feature_extraction->sequence_modeling->prediction
    base on the paper:What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis.
    https://arxiv.org/pdf/1904.01906.pdf

      改变AdaptiveAvgPool2d，使得输入tensor可以不用fixed 宽度（测试时候 不固定宽度，训练时候固定train为32*100
    TODO:train dataset 中记录iter次数，然后超过一定次数 改变图像的长宽比例，这样就可以实现mutil scale train 了
    """

    def __init__(
            self,
            backbone,
            sequence=None,
            transformation = None,
            label_head = None,
            train_cfg = None,
            test_cfg = None,
            pretrained = None,
            max_len_labels=25):
        super(FourStageRecoModel,self).__init__()
        self.backbone = build_backbone(backbone)
        self.sequenceModeling = build_sequence(sequence)
        self.img_transformation = None
        if transformation !=None:
            self.img_transformation = build_img_trans(transformation)
        self.label_head = build_head(label_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # max_len_labels+1 encoder num_steps
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
            (1, max_len_labels+1))  # Transform final w->max_len_labels  (imgH/16-1) -> 1
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        super(FourStageRecoModel,self).init_weights(pretrained)
        if self.img_transformation !=None:
            self.img_transformation.init_weights()
        self.backbone.init_weights(pretrained)
        self.sequenceModeling.init_weights()
    def extract_feat(self, x):
        """Directly extract features from the img backbone
        """
        """ Transformation stage """
        if self.img_transformation:
            x = self.img_transformation(x)

        """ Feature extraction stage """
        #512 x 1 x 26[c,h,w]
        visual_feature = self.backbone(x)
        visual_feature = self.AdaptiveAvgPool(visual_feature)

        # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.permute(0,3,1,2)
        visual_feature = visual_feature.squeeze(3)

        contextual_feature = self.sequenceModeling(visual_feature)
        return contextual_feature



    def forward_train(self,
                      data:dict,
                      **kwargs):
        img = data.get("img")
        x = self.extract_feat(img)
        data['img'] = x
        outs = self.label_head(data,return_loss=True)
        loss_inputs = outs
        losses = self.label_head.loss(*loss_inputs)
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














