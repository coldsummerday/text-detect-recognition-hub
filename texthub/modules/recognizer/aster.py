
import torch.nn as nn
from .base import  BaseRecognizer
from ..registry import RECOGNIZERS
from ..builder import build_backbone,build_img_trans,build_sequence,build_head


@RECOGNIZERS.register_module
class AsterRecognizer(BaseRecognizer):
    def __init__(self,
                 backbone,
                 sequence,
                 label_head,
                 transformation = None,
                 train_cfg = None,
                test_cfg = None,
                pretrained = None):
        super(AsterRecognizer, self).__init__()
        self.backbone = build_backbone(backbone)
        self.sequence = build_sequence(sequence)
        self.img_transformation = None
        if transformation != None:
            self.img_transformation = build_img_trans(transformation)
        self.label_head = build_head(label_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.encoder = AsterEncoder(self.backbone,self.img_transformation,self.sequence)
        self.init_weights(pretrained)

    def extract_feat(self,data):
        return  self.encoder(data)


    def init_weights(self, pretrained=None):
        super(AsterRecognizer,self).init_weights(pretrained)
        if self.img_transformation !=None:
            self.img_transformation.init_weights()
        self.backbone.init_weights(pretrained)
        self.sequence.init_weights()

    def forward(self, data, return_loss=True, **kwargs):
        outputs =None
        if return_loss:
            outputs = self.forward_train(data, **kwargs)
        else:
            outputs =  self.forward_test(data, **kwargs)
        return  outputs

    def forward_train(self, data):
        data = self.extract_feat(data)
        ## [batch,max_len+1,num_class]  with '[s]'
        probs = self.label_head(data,return_loss=True)

        ## [batch,max_len+2]  with '[go],[s]'
        target = data.get('label')
        ## ignore the first '[go]'
        loss_inputs = {"probs":probs,"target":target[:,1:]}
        losses = self.label_head.loss(**loss_inputs)
        return losses


    def forward_test(self,data):
        data = self.extract_feat(data)
        outs = self.label_head(data,return_loss=False)
        return outs
    def postprocess(self,data):
        return self.label_head.postprocess(data)


        


class AsterEncoder(nn.Module):
    def __init__(self,backbone:nn.Module,transformation:nn.Module,sequence:nn.Module):
        super(AsterEncoder, self).__init__()
        self.backbone = backbone
        self.tps = transformation
        self.sequence = sequence
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

    def forward(self,data:dict):
        x = data.get("img")

        """ Transformation stage """
        if self.tps:
            x = self.tps(x)

        """ Feature extraction stage """
        # extract visual feature
        x = self.backbone(x)

        x = self.AdaptiveAvgPool(x.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        # translate visual feature to contextual feature
        ##(b,w,c*h) - > (b,seq_len,seq_dim)
        x = x.squeeze(3)

        x = self.sequence(x)
        data["img"] = x
        return data


