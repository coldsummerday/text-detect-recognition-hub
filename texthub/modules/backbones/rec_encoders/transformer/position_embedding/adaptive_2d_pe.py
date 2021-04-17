import torch.nn as nn
import torch

from .base_pe_ggenerate import generate_position_embedding


class Adaptive2DPositionEncoder(nn.Module):
    """

    """
    def __init__(self,in_channels:int,max_h:int=200,max_w:int=200,dropout=0.1):
        super(Adaptive2DPositionEncoder, self).__init__()

        h_position_embedding = generate_position_embedding(in_channels,max_h)#[max_len,in_channels]
        #max-len,in_channels-> [1,in_channels,max_len,1]
        h_position_embedding = h_position_embedding.transpose(0, 1).view(1, in_channels, max_h, 1)


        w_position_embedding = generate_position_embedding(in_channels,max_w)
        w_position_embedding = w_position_embedding.transpose(0,1).view(1,in_channels,1,max_w)

        """
        register_buffer 作用:h_position_embedding并非module参数，但是需要作为model状态的一部分
        """
        self.register_buffer("h_position_embedding",h_position_embedding)
        self.register_buffer("w_position_embedding",w_position_embedding)

        self.h_scale_layer = self.scale_factor_generate(in_channels)
        self.w_scale_layer = self.scale_factor_generate(in_channels)

        self.pool_layer = nn.AdaptiveMaxPool2d(1)
        self.dropout_layer = nn.Dropout(p=dropout)


    def scale_factor_generate(self,in_channels:int):
        ##生成对w，h的权重系数
        scale_factor = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,in_channels,kernel_size=1),
            nn.Sigmoid()
        )
        return scale_factor

    def forward(self,x:torch.Tensor):
        b,c,h,w = x.size()
        avg_pool = self.pool_layer(x)

        h_pos_embedding = self.h_scale_layer(avg_pool) * self.h_position_embedding[:, :, :h, :]
        w_pos_embedding = self.w_scale_layer(avg_pool) * self.w_position_embedding[:, :, :, :w]

        out = x + h_pos_embedding + w_pos_embedding
        return out