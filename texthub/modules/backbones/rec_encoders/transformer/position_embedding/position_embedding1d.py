import  torch.nn as nn
import torch
from .base_pe_ggenerate import generate_position_embedding


class PositionEncoder1D(nn.Module):
    def __init__(self,in_channels:int,max_len:int=2000,dropout:float=0.1):
        super(PositionEncoder1D, self).__init__()
        position_encoder = generate_position_embedding(in_channels,max_len)
        position_encoder = position_encoder.unsqueeze(0)
        self.register_buffer('position_encoder', position_encoder)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x:torch.Tensor):
        visual_flag = False
        if len(x.shape)==4:
            visual_flag = True
            b, c, h, w = x.size()
            x = x.view(b, c, h * w).transpose(1, 2)


        #原始PE
        out = x + self.position_encoder[:,:x.size(1),:]
        out = self.dropout(out)



        if visual_flag:
            out = out.transpose(1, 2).contiguous().view(b, c, h, w)
        return out