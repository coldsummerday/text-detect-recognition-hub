from  .botnet import BottleStack

import  torch.nn as nn
from .baseresent import BasicBlock
from  ..utils.moduleinit import kaiming_init,normal_init,constant_init
from ..registry import BACKBONES

def gnnorm2d(num_channels, num_groups=32):
    if num_groups > 0:
        return nn.GroupNorm(num_groups,num_channels)
    else:
        return nn.BatchNorm2d(num_channels)



@BACKBONES.register_module
class AsterBotResNet(nn.Module):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """
    def __init__(self, input_channel,input_size:tuple ,output_channel=512,gn=False,heads:int=4,dim_head:int=128*2,proj_factor:int=4):
        super(AsterBotResNet, self).__init__()
        if gn==True:
            self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3],norm_layer=gnnorm2d)
        else:
            self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3], norm_layer=None)
        ##只改变C5
        img_h,img_w = input_size

        #c4
        self.ConvNet.layer3=BottleStack(
            dim=256,
            fmap_size=(img_h // 8, img_w // 4 + 1),  # feature map size (img_h//8, img_w//4+1)
            dim_out=512,  # channels out
            proj_factor=proj_factor,  # projection factor
            downsample=False,  # downsample on first layer or not
            heads=heads,  # number of heads
            dim_head=dim_head,  # dimension per head, defaults to 128
            rel_pos_emb=True,  # use relative positional embedding - uses absolute if False
            activation=nn.ReLU()  # activation throughout the network,
        )

        #c5
        self.ConvNet.layer4 =  BottleStack(
            dim = 512,              # channels in
            fmap_size = (img_h//8,img_w//4+1),     # feature map size (img_h//8, img_w//4+1)
            dim_out = output_channel, # channels out
            proj_factor = proj_factor,        # projection factor
            downsample = False,      # downsample on first layer or not
            heads = heads,              # number of heads
            dim_head = dim_head,         # dimension per head, defaults to 128
            rel_pos_emb = True,    # use relative positional embedding - uses absolute if False
            activation = nn.ReLU()  # activation throughout the network,

            )

    def forward(self, input):
        return self.ConvNet(input)

    def init_weights(self,pretrained=None):
        if pretrained is None:
            for m in self.ConvNet.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
                ##TODO:GroupNorm init
        elif isinstance(pretrained,str):
            ##TODO:load pretrain model from pth
            pass
        else:
            raise TypeError('pretrained must be a str or None')


class ResNet(nn.Module):
    def __init__(self, input_channel, output_channel, block, layers,norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = self._norm_layer(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[
                               0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[
                               1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = self._norm_layer(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[
                               2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = self._norm_layer(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)

        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                 3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = self._norm_layer(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                 3], kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = self._norm_layer(self.output_channel_block[3])


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self._norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)


        x = self.layer4(x)

        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x