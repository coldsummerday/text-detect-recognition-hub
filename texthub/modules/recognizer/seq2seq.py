import torch.nn as nn
from .base import  BaseRecognizer
from ..registry import RECOGNIZERS
from ..builder import build_backbone,build_img_trans,build_sequence,build_head



@RECOGNIZERS.register_module
class Seq2SeqAttention(BaseRecognizer):
    """
    seq2seq model for image Recognizer 
    """
    def __init__(
            self,
            backbone,
            feature_wh = 24,
            batch_max_length = 25,
            transformation = None,
            label_head = None,
            train_cfg = None,
            test_cfg = None,
            pretrained = None):
        super(Seq2SeqAttention,self).__init__()
        self.backbone = build_backbone(backbone)
        self.img_transformation = None
        if transformation !=None:
            self.img_transformation = build_img_trans(transformation)
        self.label_head = build_head(label_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
        self.wh2seqlen = nn.Linear(feature_wh, batch_max_length+1) #Transform w*h to be seq_len dim
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        super(Seq2SeqAttention,self).init_weights(pretrained)
        if self.img_transformation !=None:
            self.img_transformation.init_weights()
        self.backbone.init_weights(pretrained)


    def extract_feat(self, x):
        """Directly extract features from the img backbone
        """

        batch_size = x.size(0)
        """ Transformation stage """
        if self.img_transformation:
            x = self.img_transformation(x)

        """ Feature extraction stage """
        visual_feature = self.backbone(x)
        # visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        # visual_feature = visual_feature.squeeze(3)

        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 1, 3, 2))  # [b, c, h, w] -> [b,c,w,h]
        channel_size = visual_feature.size(1)
        visual_feature = visual_feature.squeeze(3)  # [b,c,w*h]
        visual_feature = self.wh2seqlen(visual_feature) #[b,c,w*h]-[b,c,seq_len]
        # [b,w*h,c] -> [b,w*h as  seq_len,c as inputdimentions]
        visual_feature = visual_feature.view(batch_size, -1, channel_size)
        return visual_feature

    def forward_train(self,
                      data:dict,
                      **kwargs):
        img = data.get("img")
        x = self.extract_feat(img)
        data['img'] = x
        loss_inputs = self.label_head(data,return_loss=True)
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














