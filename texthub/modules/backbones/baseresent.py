import torch.nn as nn
'''
2020 07 23  changelog:basicblock模块添加dcn
'''
import  torchvision.models.resnet


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,with_dcn:bool=False,dcn_config:dict=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.with_dcn = with_dcn
        self.dcn_config = dcn_config
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if not self.with_dcn:
            self.conv2 = conv3x3(planes, planes)
        else:
            self.with_modulated_dcn = self.dcn_config.get('modulated', False)
            deformable_groups = self.dcn_config.get('deformable_groups', 1)
            self.with_modulated_dcn = self.dcn_config.get('modulated', False)
            if not self.with_modulated_dcn:
                from texthub.ops.dcn import DeformConv
                conv_op = DeformConv
                offset_channels = 18
            else:
                from texthub.ops.dcn import ModulatedDeformConv
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                padding=1)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                padding=1,
                deformable_groups=deformable_groups,
                bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

##TODO:BUG 是否downsample条件出错
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,with_dcn:bool=False,dcn_config:dict=None):
        super(Bottleneck, self).__init__()
        self.with_dcn = with_dcn
        self.dcn_config = dcn_config
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        if not self.with_dcn:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        else:
            self.with_modulated_dcn = self.dcn_config.get('modulated', False)
            deformable_groups = self.dcn_config.get('deformable_groups', 1)
            self.with_modulated_dcn = self.dcn_config.get('modulated', False)
            if not self.with_modulated_dcn:
                from texthub.ops.dcn import DeformConv
                conv_op = DeformConv
                offset_channels = 18
            else:
                from texthub.ops.dcn import ModulatedDeformConv
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                width,
                deformable_groups * offset_channels,
                kernel_size=3,
                padding=1)
            self.conv2 = conv_op(
                width,
                width,
                kernel_size=3,
                padding=1,
                stride=stride,
                deformable_groups=deformable_groups,
                bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out