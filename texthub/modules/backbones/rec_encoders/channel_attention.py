import  torch
import  torch.nn as  nn
from  ...utils.moduleinit import kaiming_init,constant_init
valid_fusion_types = ['channel_add', 'channel_mul']


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

"""
视觉上下文信息模块，采取attention机制进行信息mask融合
"""
class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes:int,
                 ratio:float,
                 pooling_type='att',
                 fusion_types=('channel_mul', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x:torch.Tensor):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x:torch.Tensor):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

# class ContextBlock(nn.Module):
#     def __init__(self,
#                  inchannels:int,
#                  ratio:float,
#                  pooling_type:str="attn",
#                  fusion_types=("channel_mul",)):
#         super(ContextBlock, self).__init__()
#         assert pooling_type in ["attn","avg"]
#         assert isinstance(fusion_types,(list,tuple))
#         assert all([f in valid_fusion_types for f in fusion_types])
#         assert len(fusion_types) > 0, 'at least one fusion should be used'
#
#         self.inchannels = inchannels
#         self.ratio = ratio
#
#         self.channels = int(inchannels * ratio)
#         self.pooling_type = pooling_type
#         self.fusion_types = fusion_types
#
#         if self.pooling_type=='attn':
#             self.conv_mask = nn.Conv2d(inchannels,1,kernel_size=1)
#             self.softmask = nn.Softmax(dim=2)
#         else:
#             self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#         if 'channel_add' in fusion_types:
#             self.channel_add_conv = nn.Sequential(
#                 nn.Conv2d(self.inchannels,self.channels,kernel_size=1),
#                 nn.LayerNorm([self.channels,1,1]),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(self.channels,self.inchannels,kernel_size=1)
#             )
#         else:
#             self.channel_add_conv = None
#
#         if 'channel_mul' in fusion_types:
#             self.channel_mul_conv = nn.Sequential(
#                 nn.Conv2d(self.inchannels, self.channels, kernel_size=1),
#                 nn.LayerNorm([self.channels, 1, 1]),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(self.channels, self.inchannels, kernel_size=1)
#             )
#         else:
#             self.channel_mul_conv = None
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         if self.pooling_type == 'attn':
#             kaiming_init(self.conv_mask, mode='fan_in')
#             self.conv_mask.inited = True
#
#         if self.channel_add_conv is not None:
#             last_zero_init(self.channel_add_conv)
#         if self.channel_mul_conv is not None:
#             last_zero_init(self.channel_mul_conv)
#
#     def spatial_pool(self,x:torch.Tensor):
#         batch,channle,height,width = x.size()
#         if self.pooling_type=="attn":
#             input_x = x
#             # # [N, C, H * W]
#             # input_x = input_x.view(batch,channle,height * width)
#             # # # [N, 1, C, H * W]
#             # # input_x = input_x.unsqueeze(1)
#             # [N, 1, H, W]
#             context_mask = self.conv_mask(input_x)
#             # [N, 1, H * W]
#             context_mask = context_mask.view(batch,1,height * width)
#             # [N, 1, H * W]
#             context_mask = self.softmask(context_mask)
#             # [N, 1, H * W, 1]
#             context_mask = context_mask.unsqueeze(-1)
#             # [N, 1, C, 1]
#             context = torch.matmul(input_x,context_mask)
#             # [N, C, 1, 1]
#             context = context.view(batch,channle,1,1)
#         else:
#             # [N, C, 1, 1]
#             context = self.avg_pool(x)
#         return context
#     def forward(self,x:torch.Tensor):
#
#         #[N,C,1,1]
#         context = self.spatial_pool(x)
#         out = x
#
#         if self.channel_mul_conv is not None:
#             #[N,C,1,1]
#             channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
#             out = out * channel_mul_term
#         if self.channel_add_conv is not None:
#             #[N,C,1,1]
#             channel_add_term = self.channel_add_conv(context)
#             out = out + channel_add_term
#         return out

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)
