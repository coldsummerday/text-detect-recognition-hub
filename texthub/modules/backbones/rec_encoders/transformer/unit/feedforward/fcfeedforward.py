import torch
import torch.nn as nn

class FCFeedforward(nn.Module):
    def __init__(self,hidden_dim:int,dropout:float=0.1):
        super(FCFeedforward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=hidden_dim,out_features=hidden_dim*4,bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim*4,out_features=hidden_dim,bias=True),
            nn.Dropout(p=dropout)

        )

    def forward(self, x):
        out = self.layers(x)
        return out

"""
  feedforward=dict(
                        type='Feedforward',
                        layers=[
                            dict(type='FCModule', in_channels=hidden_dim, out_channels=hidden_dim * 4, bias=True,
                                 activation='relu', dropout=dropout),
                            dict(type='FCModule', in_channels=hidden_dim * 4, out_channels=hidden_dim, bias=True,
                                 activation=None, dropout=dropout),
                        ],
                    ),
"""