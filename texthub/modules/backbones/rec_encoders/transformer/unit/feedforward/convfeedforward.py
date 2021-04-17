

import torch
import torch.nn as nn

class ConvFeedforward(nn.Module):
    def __init__(self,hidden_dim:int,dropout:float=0.1):
        super(ConvFeedforward, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim,out_channels=hidden_dim*4,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=hidden_dim*4, out_channels=hidden_dim * 1, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=dropout)

        )

    def forward(self, x):
        out = self.layers(x)
        return out
