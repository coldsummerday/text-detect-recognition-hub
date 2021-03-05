import torch
import torch.nn as nn
from texthub.modules.registry import RECOGNIZERS
from texthub.modules.builder import build_backbone,build_img_trans,build_head

from texthub.modules.backbones.rec_encoders import MeanShift


@RECOGNIZERS.register_module
class PlugNet(nn.Module):
    """
    PlugNet: Degradation Aware Scene Text Recognition Supervised by a Pluggable Super-Resolution Unit

    利用features层次 做sr，更好地训练backbone网络对低分辨率的识别能力

    """
    def __init__(
            self,
            backbone,
            transformation=None,
            label_head=None,
            pretrained=None,
            **kwargs
    ):
        super(PlugNet, self).__init__()
        self.backbone = build_backbone(backbone)
        self.img_transformation = None
        if transformation != None:
            self.img_transformation = build_img_trans(transformation)
        self.label_head = build_head(label_head)
        self.init_weights(pretrained)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(255, rgb_mean, rgb_std)

    def init_weights(self, pretrained=None):
        if self.img_transformation != None:
            self.img_transformation.init_weights()
        self.backbone.init_weights(pretrained)



    def forward_train(self,
                      data: dict,
                      **kwargs):
        """
        'hr_img','lr_img'
        """
        lr_img, hr_img =data.get("lr_img"),data.get("img")

        if self.img_transformation:
            lr_img = self.img_transformation(lr_img)
            hr_img = self.img_transformation(hr_img)

        x = self.sub_mean(lr_img)


        encoder_feats, sharing_feats = self.backbone(x)
        encoder_feats = encoder_feats.contiguous()

        data["encoder_feats"] = encoder_feats
        data["sharing_feats"] = sharing_feats
        # data["lr_img"] = lr_img
        data["img"] = hr_img

        losses = self.label_head(data, return_loss=True)
        ##在runner中loss 回传
        return losses

    def forward_test(self, data: dict):

        """
            'hr_img','lr_img'
        """
        hr_img =data.get("img")
        # lr_img = data.get("lr_img")
        if self.img_transformation:
            # lr_img = self.img_transformation(lr_img)
            hr_img = self.img_transformation(hr_img)

        ##用ori_img做前馈
        x = self.sub_mean(hr_img)
        encoder_feats, _ = self.backbone(x)
        encoder_feats = encoder_feats.contiguous()

        data['encoder_feats'] = encoder_feats
        outs = self.label_head(data, return_loss=False)
        return outs

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












