import torch
from torch import nn
import torch.nn.functional as F
from ..registry import NECKS

@NECKS.register_module
class PseFPN(nn.Module):
    def __init__(self,input_channels:[int]=[256, 512, 1024, 2048],conv_out = 256,inplace = True):
        super(PseFPN, self).__init__()
        self.conv_out = conv_out
        # Reduce channels
        # Top layer
        self.toplayer = nn.Sequential(nn.Conv2d(input_channels[3], conv_out, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(conv_out),
                                      nn.ReLU(inplace=inplace)
                                      )
        # Lateral layers
        self.latlayer1 = nn.Sequential(nn.Conv2d(input_channels[2], conv_out, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(conv_out),
                                       nn.ReLU(inplace=inplace)
                                       )
        self.latlayer2 = nn.Sequential(nn.Conv2d(input_channels[1], conv_out, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(conv_out),
                                       nn.ReLU(inplace=inplace)
                                       )
        self.latlayer3 = nn.Sequential(nn.Conv2d(input_channels[0], conv_out, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(conv_out),
                                       nn.ReLU(inplace=inplace)
                                       )

        # Smooth layers
        self.smooth1 = nn.Sequential(nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(conv_out),
                                     nn.ReLU(inplace=inplace)
                                     )
        self.smooth2 = nn.Sequential(nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(conv_out),
                                     nn.ReLU(inplace=inplace)
                                     )
        self.smooth3 = nn.Sequential(nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(conv_out),
                                     nn.ReLU(inplace=inplace)
                                     )

        # self.conv = nn.Sequential(
        #     nn.Conv2d(conv_out * 4, conv_out, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(conv_out),
        #     nn.ReLU(inplace=inplace)
        # )
        # self.out_conv = nn.Conv2d(conv_out, result_num, kernel_size=1, stride=1)

    def forward(self, features: [torch.Tensor]):
        c2, c3, c4, c5 = features
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.smooth3(p2)

        x = self._upsample_cat(p2, p3, p4, p5)
        return x

        # x = self.conv(x)
        # x = self.out_conv(x)
        #
        # if self.train:
        #     x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        # else:
        #     x = F.interpolate(x, size=(H // self.scale, W // self.scale), mode='bilinear', align_corners=True)
        # return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=False) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=False)
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear', align_corners=False)
        return torch.cat([p2, p3, p4, p5], dim=1)
