import torch


def generate_position_embedding(in_channels:int,max_len:int)->torch.Tensor:
    pos = torch.arange(max_len).float().unsqueeze(1) ##[max_len,1]
    index_tensor = torch.arange(in_channels).float().unsqueeze(0) ##[1,in_channels]
    angle_rate = 1/torch.pow(10000,(2*(index_tensor//2)/in_channels))
    position_encoder = pos * angle_rate
    position_encoder[:,0::2] = torch.sin(position_encoder[:,0::2])
    position_encoder[:,1::2] = torch.cos(position_encoder[:,1::2])
    ##[mac_len,in_channels]
    return position_encoder

