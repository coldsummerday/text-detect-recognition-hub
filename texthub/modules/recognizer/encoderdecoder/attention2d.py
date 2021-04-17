
import torch
import torch.nn as nn
from texthub.modules.registry import RECOGNIZERS
from texthub.modules.builder import build_backbone,build_img_trans,build_head


@RECOGNIZERS.register_module
class Satrn(nn.Module):
    """
    paper name:
    On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention
    Self-Attention Text Recognition Network

    transformer encoder  -decoder


    """
    def __init__(
            self,
            backbone,
            transformation=None,
            label_head=None,
            pretrained=None,
            **kwargs
    ):
        super(Satrn, self).__init__()
        self.backbone = build_backbone(backbone)
        self.img_transformation = None
        if transformation != None:
            self.img_transformation = build_img_trans(transformation)
        self.label_head = build_head(label_head)
        self.init_weights(pretrained)



    def init_weights(self, pretrained=None):
        if self.img_transformation != None:
            self.img_transformation.init_weights()
        self.backbone.init_weights(pretrained)



    def forward_train(self,
                      data: dict,
                      **kwargs):

        conv_features = data.get('img')
        if self.img_transformation:
            conv_features = self.img_transformation(conv_features)

        encoder_features = self.backbone(conv_features)
        data["img"] = encoder_features
        losses = self.label_head(data, return_loss=True)
        ##在runner中loss 回传
        return losses

    def forward_test(self, data: dict):
        conv_features = data.get('img')
        if self.img_transformation:
            conv_features = self.img_transformation(conv_features)

        encoder_features = self.backbone(conv_features)
        data["img"] = encoder_features
        preds = self.label_head(data, return_loss=False)
        return preds

    def postprocess(self, preds:torch.Tensor):
        return self.label_head.postprocess(preds)

    def forward(self, data, return_loss=True):
        """
        in the text recognition ,the extra_data is the label
        """
        outputs = None
        if return_loss:
            outputs = self.forward_train(data)

        else:
            outputs = self.forward_test(data)

        return outputs






