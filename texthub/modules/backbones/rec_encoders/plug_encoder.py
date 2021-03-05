import  torch
import  torch.nn as  nn
from ..baseresent import BasicBlock
from .channel_attention import ContextBlock
from ...registry import BACKBONES

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

@BACKBONES.register_module
class ResNetLstm_Plug(nn.Module):
    def __init__(self,n_group=1,in_channels=3,hidden_dim= 512):
        super(ResNetLstm_Plug, self).__init__()
        self.n_group = n_group

        in_channels = in_channels
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels,32,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        #[n,c,32,100]
        self.inplanes = 32
        self.layer1 = self._make_layer(BasicBlock,32, 3, 2)  # [16, 50]
        self.layer2 = self._make_layer(BasicBlock,64, 4,  2)  # [8, 25]

        ## feature enhanced
        self.layer3 = self._make_layer(BasicBlock,128, 6,  1)  # [8, 25]
        self.layer4 = self._make_layer(BasicBlock,256, 6,  1)  # [8, 25]
        self.layer5 = self._make_layer(BasicBlock,512, 3,  1)  # [8, 25]

        ##concate 3,4,5   128+256+512 = 896
        self.inplanes = 896
        self.gc_block = ContextBlock(self.inplanes, ratio=0.8)
        self.layer6 = self._make_layer(BasicBlock,128, 1,  1)  # [8, 25, 128]


        self.rnn = nn.LSTM(8*128,hidden_dim,bidirectional=True,num_layers=2,batch_first=True)
        self.outplanes = 8*128

        self._flattened =  False

    def init_weights(self,pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x:torch.Tensor):

        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        #feature concate
        x_concate = torch.cat([x3,x4,x5],dim=1)
        gc_block = self.gc_block(x_concate)
        # [8, 25, 128]
        x6 = self.layer6(gc_block)

        #cnn_feature ->rnn_feature
        #[b,h,w,c] - > [b,c,h,w]
        cnn_feat = x6.permute(0,3,1,2)
        #[b,c,h*w]
        cnn_feat = cnn_feat.view(cnn_feat.size(0), cnn_feat.size(1), -1)

        if not self._flattened:
            self.rnn.flatten_parameters()
            self._flattened = True
        rnn_feat, _ = self.rnn(cnn_feat)
        #encoder_feats, sharing_feats
        return rnn_feat,x5



    def _make_layer(self, block:nn.Module, planes:int, blocks:int, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNet34Lstm_Plug(nn.Module):
    def __init__(self,n_group=1,in_channels=3,hidden_dim= 512):
        super(ResNet34Lstm_Plug, self).__init__()
        self.n_group = n_group

        in_channels = in_channels
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,32,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        #[n,c,32,100]
        self.inplanes = 32
        self.layer0 = self._make_layer(BasicBlock,32, 3, 2)  # [16, 50]
        self.layer1 = self._make_layer(BasicBlock,64, 3,  2)  # [8, 25]

        ## feature enhanced
        self.layer2 = self._make_layer(BasicBlock,128, 4,  1)  # [8, 25]
        self.layer3 = self._make_layer(BasicBlock,256, 6,  1)  # [8, 25]
        self.layer4 = self._make_layer(BasicBlock,512, 3,  1)  # [8, 25]

        ##concate 2,3,4   128+256+512 = 896
        self.inplanes = 896
        self.gc_block = ContextBlock(self.inplanes, ratio=0.8)
        self.layer5 = self._make_layer(BasicBlock,128, 1,  1)  # [8, 25, 128]


        self.rnn = nn.LSTM(8*128,hidden_dim,bidirectional=True,num_layers=2,batch_first=True)
        self.outplanes = 8*128

        self._flattened =  False

    def init_weights(self,pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        resnet34_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
        state_dict = load_state_dict_from_url(resnet34_url)
        try:
            self.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            ###layer 1 中第一个conv resnet34 是64->64，而本代码中是32-64，所以忽略该错误
            pass

    def forward(self,x:torch.Tensor):

        x0 = self.layer(x)
        x1 = self.layer0(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        #feature concate
        x_concate = torch.cat([x3,x4,x5],dim=1)
        gc_block = self.gc_block(x_concate)
        # [8, 25, 128]
        x6 = self.layer5(gc_block)

        #cnn_feature ->rnn_feature
        #[b,h,w,c] - > [b,c,h,w]
        cnn_feat = x6.permute(0,3,1,2)
        #[b,c,h*w]
        cnn_feat = cnn_feat.view(cnn_feat.size(0), cnn_feat.size(1), -1)

        if not self._flattened:
            self.rnn.flatten_parameters()
            self._flattened = True
        rnn_feat, _ = self.rnn(cnn_feat)
        #encoder_feats, sharing_feats
        return rnn_feat,x5



    def _make_layer(self, block:nn.Module, planes:int, blocks:int, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


def position_embedding(n_position:int,feat_dim:int,wave_length=10000):

    # [n_position]
    positions = torch.arange(0,n_position)

    dim_range = torch.arange(0,feat_dim)
    dim_range = torch.pow(wave_length,2*(dim_range//2)/feat_dim)
    # [n_position, feat_dim]
    angles = positions.unsqueeze(1) / dim_range.unsqueeze(0)
    angles = angles.float()

    angles[:,0::2] = torch.sin(angles[:,0::2])
    angles[:,1::2] = torch.cos(angles[:,1::2])
    return angles


if __name__ == "__main__":
  x = torch.randn(3, 3, 32, 100)
  net = ResNetLstm_Plug()
  encoder_feat = net(x)
  print(encoder_feat.size())


# class PlugNetEncoder(nn.Module):
#     def __init__(self):
#         super(PlugNetEncoder, self).__init__()
#


