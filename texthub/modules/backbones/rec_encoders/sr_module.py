import torch.nn as  nn
import torch
import math
class CALayer(nn.Module):
    def __init__(self,channels,reduction:int=16):
        super(CALayer, self).__init__()

        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels,channels//reduction,1,padding=0,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction,channels,1,padding=0,bias=True),
            nn.Sigmoid()
        )
    def forward(self,x:torch.Tensor):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self,
                 conv:nn.Module,n_feat:int,kernel_size:int,reduction:int,bias=True,bn=False,act_fun = nn.ReLU(inplace=True),res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat,n_feat,kernel_size,bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i==0:
                modules_body.append(act_fun)
        modules_body.append(CALayer(n_feat,reduction=reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self,x:torch.Tensor):
        res = self.body(x)
        return res+x

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction,n_resblocks,act=nn.ReLU(inplace=True), res_scale=1):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act_fun=act, res_scale=res_scale) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    """
    将[n,512,h/4,w/4]图像恢复至[n,3,h,w]
    """
    def __init__(self,in_channels = 512,mid_channels=64,out_channels=3,scale=4,num_resgroups=2,num_resblocks=2,conv=default_conv):
        super(RCAN, self).__init__()

        num_feats = mid_channels
        kernel_size = 3
        reduction = 16
        action_fun = nn.ReLU(inplace=True)

        # modules_body = [
        #     ResidualGroup(
        #         conv, num_feats, kernel_size, reduction, act=action_fun, res_scale=1, n_resblocks=num_resblocks) \
        #     for _ in range(num_resgroups)]
        # modules_body.append(conv(num_feats, num_feats, kernel_size))

        modules_body = []
        for _ in range(num_resgroups):
            modules_body.append(ResidualGroup(conv=conv,n_feat=num_feats,kernel_size=kernel_size,reduction=reduction,act=action_fun,res_scale=1,n_resblocks=num_resblocks))
        modules_body.append(conv(num_feats,num_feats,kernel_size=kernel_size))

        module_tail = [
            Upsampler(conv=conv,scale=scale,n_feat=num_feats,act=False),
            conv(num_feats,out_channels,kernel_size=kernel_size)
        ]

        self.conv_1 = conv(in_channels=in_channels,out_channels=num_feats,kernel_size=kernel_size)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*module_tail)

    def forward(self,x:torch.Tensor):
        x = self.conv_1(x)
        res = self.body(x)
        res +=x

        x = self.tail(res)
        return x

