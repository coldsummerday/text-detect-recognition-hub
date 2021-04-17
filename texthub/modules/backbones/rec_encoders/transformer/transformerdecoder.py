import torch.nn as nn
import torch

from .default_units import default_attention,default_feedforward,default_position_embedding,attention_class_dict,feedforward_class_dict,position_embedding_dict,build_norm
import copy

class TransformerDecoderLayer1D(nn.Module):
    def __init__(self,
                 self_attention_dict:dict,
                 self_attention_norm_dict:dict,
                 attention_dict:dict,
                 attention_norm_dict:dict,
                 feedforward_dict:dict,
                 feedforward_norm_dict:dict):
        super(TransformerDecoderLayer1D, self).__init__()
        if "type" in self_attention_dict.keys():
            self_attention_class = attention_class_dict[self_attention_dict.pop("type")]
        else:
            self_attention_class = default_attention


        if "type" in attention_dict.keys():
            attention_class = attention_class_dict[attention_dict.pop("type")]
        else:
            attention_class = default_attention

        if "type" in feedforward_dict.keys():
            feedforward_class = feedforward_class_dict[feedforward_dict.pop("type")]
        else:
            feedforward_class = default_feedforward

        self.self_attention_layer = self_attention_class(**self_attention_dict)
        self.self_attention_norm_layer = build_norm(self_attention_norm_dict)

        self.attention_layer = attention_class(**attention_dict)
        self.attention_norm_layer = build_norm(attention_norm_dict)

        self.feedforward_layer = feedforward_class(**feedforward_dict)
        self.feedforward_norm_layer = build_norm(feedforward_norm_dict)

    def forward(self,tgt:torch.Tensor,src:torch.Tensor,tgt_mask=None,src_mask=None):

        attn1,_ = self.self_attention_layer(tgt,tgt,tgt,tgt_mask)
        out1 = self.self_attention_norm_layer(tgt+attn1)

        size = src.size()
        if len(size)==4:
            b,c,h,w = size
            src = src.view(b,c,h*w).transpose(1,2)
            if src_mask is not None:
                src_mask = src_mask.view(b,1,h*w)

        attn2,_ = self.attention_layer(out1,src,src,src_mask)
        out2 = self.attention_norm_layer(out1+attn2)

        ffn_out = self.feedforward_layer(out2)
        out3 = self.feedforward_norm_layer(out2+ffn_out)
        return out3


transformer_decoder_layer_dict = {"TransformerDecoderLayer1D":TransformerDecoderLayer1D}
default_decoder_layer = TransformerDecoderLayer1D
class TransformerDecoder(nn.Module):
    def __init__(self,
                 decoder_layer_dict: dict, 
                 num_layers: int,
                 pe_dict: dict = None):
        super(TransformerDecoder, self).__init__()
        if pe_dict != None:
            if "type" in pe_dict.keys():
                pos_embedding_class = position_embedding_dict[pe_dict.pop("type")]
            else:
                pos_embedding_class = default_position_embedding
        if "type" in decoder_layer_dict.keys():
            decoder_class = transformer_decoder_layer_dict[decoder_layer_dict.pop("type")]
        else:
            decoder_class = default_decoder_layer

        self.pos_embedding_layer = pos_embedding_class(**pe_dict)

        ##在这个过程中会pop ("type"),影响下一个decoder的初始化
        self.layers = nn.ModuleList([decoder_class(**copy.deepcopy(decoder_layer_dict)) for _ in range(num_layers)])

    @property
    def with_pe(self):
        return hasattr(self, "pos_embedding_layer") and self.pos_embedding_layer is not None

    def forward(self, tgt:torch.Tensor, src:torch.Tensor, tgt_mask=None, src_mask=None):
        if self.with_pe:
            tgt = self.pos_embedding_layer(tgt)

        for layer in self.layers:
            tgt = layer(tgt, src, tgt_mask, src_mask)

        return tgt
