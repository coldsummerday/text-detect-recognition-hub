from texthub.modules.backbones import ResNet34Lstm_Plug
net = ResNet34Lstm_Plug()
net.init_weights()

import torch
x = torch.randn(3, 3, 32, 100)
y,z = net(x)

from texthub.modules.backbones.det_resnet import resnet18,resnet34

import torch
from texthub.modules.backbones.rec_encoders import RCAN
x = torch.randn(3, 512, 8, 25)
net = RCAN()
y = net(x)

