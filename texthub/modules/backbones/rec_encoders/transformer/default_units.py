import torch
import  torch.nn as nn

from .position_embedding import Adaptive2DPositionEncoder,PositionEncoder1D
from .unit.attention import  MultiHeadAttention,GaussianSelfAttention,Learned2DRelativeSelfAttention
from .unit.feedforward import  FCFeedforward,ConvFeedforward


default_attention = MultiHeadAttention
default_feedforward = FCFeedforward
default_position_embedding = PositionEncoder1D

attention_class_dict ={"MultiHeadAttention":MultiHeadAttention,
                       "GaussianSelfAttention":GaussianSelfAttention,
                       "Learned2DRelativeSelfAttention":Learned2DRelativeSelfAttention}
feedforward_class_dict = {"FCFeedforward":FCFeedforward,"ConvFeedforward":ConvFeedforward}
position_embedding_dict = {"Adaptive2DPositionEncoder":Adaptive2DPositionEncoder,"PositionEncoder1D":PositionEncoder1D}



def build_norm(norm_dict:dict):
    if norm_dict['type']=="BN":
        return nn.BatchNorm2d(norm_dict["dim"])
    elif norm_dict["type"]=="LN":
        return nn.LayerNorm(norm_dict["dim"])