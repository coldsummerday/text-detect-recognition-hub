"""
standard resnet from
torchvision/models/resnet.py


remove the first 7*7 conv use the 3   3*3 conv replace


change the forward to
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c2, c3, c4, c5


out channel:
backbone_dict = {'resnet18': {'models': resnet18, 'out': [64, 128, 256, 512]},
                 'resnet34': {'models': resnet34, 'out': [64, 128, 256, 512]},
                 'resnet50': {'models': resnet50, 'out': [256, 512, 1024, 2048]},
                 'resnet101': {'models': resnet101, 'out': [256, 512, 1024, 2048]},
                 'resnet152': {'models': resnet152, 'out': [256, 512, 1024, 2048]},
                 'resnext50_32x4d': {'models': resnext50_32x4d, 'out': [256, 512, 1024, 2048]},
                 'resnext101_32x8d': {'models': resnext101_32x8d, 'out': [256, 512, 1024, 2048]},
"""

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.nn as nn
from ..registry import BACKBONES
from .baseresent import conv1x1,conv3x3,BasicBlock,Bottleneck
model_urls = {
    'resnet18': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth',
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
}


def gnnorm2d(num_channels, num_groups=32):
    if num_groups > 0:
        return nn.GroupNorm(num_groups,num_channels)
    else:
        return nn.BatchNorm2d(num_channels)




class BaseFixResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3,zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,stage_with_dcn=[False,False,False,False],dcn_config:dict=None):
        super(BaseFixResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # self.inplanes = 64
        self.inplanes = 128

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group





        # self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        #replace with 3 3*3 conv

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = norm_layer(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = norm_layer(128)
        self.relu3 = nn.ReLU(inplace=True)



        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],with_dcn=stage_with_dcn[0],dcn_config=dcn_config)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],with_dcn=stage_with_dcn[1],dcn_config=dcn_config)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],with_dcn=stage_with_dcn[2],dcn_config=dcn_config)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],with_dcn=stage_with_dcn[3],dcn_config=dcn_config)




        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,with_dcn:bool=False,dcn_config:dict=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,with_dcn=with_dcn,dcn_config=dcn_config))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,with_dcn=with_dcn,dcn_config=dcn_config))

        return nn.Sequential(*layers)

    def init_weights(self,pretrained=True):
        return NotImplemented

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c2, c3, c4, c5

@BACKBONES.register_module
class DetFixResNet(BaseFixResNet):
    """
    ResNet backBone

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Normally 3.

    """
    def __init__(self,depth:int,arch:str,norm="gn",in_channels=3,zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                stage_with_dcn=[False,False,False,False],dcn_config:dict=None):
        assert depth in (18,34,50,101,152)
        assert arch in ('resnet18','resnet50','resnet101')

        arch_settings = {
            18: (BasicBlock, (2, 2, 2, 2)),
            34: (BasicBlock, (3, 4, 6, 3)),
            50: (Bottleneck, (3, 4, 6, 3)),
            101: (Bottleneck, (3, 4, 23, 3)),
            152: (Bottleneck, (3, 8, 36, 3))
        }

        self.arch_str = arch
        norm_setting = {
            "bn":nn.BatchNorm2d,
            "gn":gnnorm2d
        }
        block,layers = arch_settings.get(depth)
        """
        self, block, layers, in_channels=3,zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None
        """
        kwargs = {
            "norm_layer":norm_setting.get(norm),
            "in_channels":in_channels,
            "zero_init_residual":zero_init_residual,
            "groups":groups,
            "width_per_group":width_per_group,
            "replace_stride_with_dilation":replace_stride_with_dilation,
            "stage_with_dcn":stage_with_dcn,
            "dcn_config":dcn_config
        }
        
        super(DetFixResNet, self).__init__(block, layers, **kwargs)


    def init_weights(self,pretrained=True):
        ##从预训练中加载
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            state_dict = load_state_dict_from_url(model_urls[self.arch_str])
            self.load_state_dict(state_dict, strict=False)


