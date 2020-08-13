import torch
from torch import nn
import torch.nn.functional as F
from ..registry import NECKS

@NECKS.register_module
class SegDBNeck(nn.Module):
    def __init__(self,in_channels=[64,128,256,512],inner_channels = 256,bias=False,
                 *args, **kwargs
                 ):
        """
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        """
        
        super(SegDBNeck, self).__init__()

        self.out_channels = inner_channels//4
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, kernel_size=1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, kernel_size=1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, kernel_size=1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, kernel_size=1, bias=bias)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels,self.out_channels,kernel_size=3,padding=1,bias=bias),
            nn.Upsample(scale_factor=8,mode="nearest")
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels,self.out_channels,kernel_size=3,padding=1,bias=bias),
            nn.Upsample(scale_factor=4,mode="nearest")
        )
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels,self.out_channels,kernel_size=3,padding=1,bias=bias),
            nn.Upsample(scale_factor=2,mode="nearest")
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(inner_channels,self.out_channels,kernel_size=3,padding=1,bias=bias)
        )


        # self.adaptive = adaptive
        #
        # self.binarize.apply(self.weights_init)
        #
        # self.adaptive = adaptive
        # if adaptive:
        #     self.thresh = self._init_thresh(
        #         inner_channels, serial=serial, smooth=smooth, bias=bias)
        #     self.thresh.apply(self.weights_init)
        #
        # self.in5.apply(self.weights_init)
        # self.in4.apply(self.weights_init)
        # self.in3.apply(self.weights_init)
        # self.in2.apply(self.weights_init)
        # self.out5.apply(self.weights_init)
        # self.out4.apply(self.weights_init)
        # self.out3.apply(self.weights_init)
        # self.out2.apply(self.weights_init)

    def init_weights(self,pretrained=None):
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 1e-4)
        else:
            pass

    def forward(self,features):
        c2,c3,c4,c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4 #1/16
        out3 = self.up4(out4) + in3  #1/8
        out2 = self.up3(out3) + in2 #1/4

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        out_features = torch.cat((p5,p4,p3,p2),1)
        return out_features
        #neck 只做特征融合部分内容，特征生成

        #
        # # this is the pred module, not binarization module;
        # # We do not correct the name due to the trained model.
        # binary = self.binarize(out_features)
        # ##todo:out_features，binary，thresh 应该放到head中写
        # if self.training:
        #     result = OrderedDict(binary=binary)
        # else:
        #     return binary
        # if self.adaptive and self.training:
        #     if self.serial:
        #         fuse = torch.cat(
        #             (out_features, nn.functional.interpolate(
        #                 binary, out_features.shape[2:])), 1)
        #     thresh = self.thresh(fuse)
        #     thresh_binary = self.step_function(binary, thresh)
        #     result.update(thresh=thresh, thresh_binary=thresh_binary)
        # return result






