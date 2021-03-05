import torch
import torch.nn as nn
from ..builder import build_backbone
from ..registry import BACKBONES
@BACKBONES.register_module
class ResnetDilated(nn.Module):
    """
    将原始的resnet 改造为空洞卷积resent
    layer3 ，layer4 部分，提高最后两层输出的分辨率
    """
    def __init__(self, orig_resnet:dict, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial
        self.orig_resnet = build_backbone(orig_resnet)
        if dilate_scale == 8:
            self.orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            self.orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            self.orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))
        self.conv1 = self.orig_resnet.conv1
        self.bn1 = self.orig_resnet.bn1
        self.relu1 = self.orig_resnet.relu1
        self.conv2 = self.orig_resnet.conv2
        self.bn2 = self.orig_resnet.bn2
        self.relu2 = self.orig_resnet.relu2
        self.conv3 = self.orig_resnet.conv3
        self.bn3 = self.orig_resnet.bn3
        self.relu3 = self.orig_resnet.relu3
        self.maxpool = self.orig_resnet.maxpool
        self.layer1 = self.orig_resnet.layer1
        self.layer2 = self.orig_resnet.layer2
        self.layer3 = self.orig_resnet.layer3
        self.layer4 = self.orig_resnet.layer4
    def init_weights(self,pretrained=None):
        self.orig_resnet.init_weights(pretrained)

        # # take pretrained resnet, except AvgPool and FC
        # self.conv1 = orig_resnet.conv1
        # self.bn1 = orig_resnet.bn1
        # self.relu1 = orig_resnet.relu1
        # self.conv2 = orig_resnet.conv2
        # self.bn2 = orig_resnet.bn2
        # self.relu2 = orig_resnet.relu2
        # self.conv3 = orig_resnet.conv3
        # self.bn3 = orig_resnet.bn3
        # self.relu3 = orig_resnet.relu3
        # self.maxpool = orig_resnet.maxpool
        # self.layer1 = orig_resnet.layer1
        # self.layer2 = orig_resnet.layer2
        # self.layer3 = orig_resnet.layer3
        # self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=True):
        # return self.orig_resnet(x)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)



        if return_feature_maps:
            return c2,c3,c4,c5
        return c5
