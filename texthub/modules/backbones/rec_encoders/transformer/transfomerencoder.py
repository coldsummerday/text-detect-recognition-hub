import torch
import torch.nn as nn
import copy

from .default_units import default_attention,default_feedforward,default_position_embedding,attention_class_dict,feedforward_class_dict,position_embedding_dict,build_norm
class TransformerEncoderLayer2D(nn.Module):
    def __init__(self,attention_dict:dict,attention_norm_dict:dict,feedforward_dict:dict,feedforward_norm_dict:dict):
        super(TransformerEncoderLayer2D, self).__init__()
        if "type" in attention_dict.keys():
            attention_class = attention_class_dict[attention_dict.pop("type")]
        else:
            attention_class = default_attention

        if "type" in feedforward_dict.keys():
            feedforward_class = feedforward_class_dict[feedforward_dict.pop("type")]
        else:
            feedforward_class = default_feedforward

        self.attention_layer = attention_class(**attention_dict)
        self.attention_norm_layer = build_norm(attention_norm_dict)
        self.feedforward_layer = feedforward_class(**feedforward_dict)
        self.feedforward_norm_layer = build_norm(feedforward_norm_dict)

    def norm_forward(self,norm_layer:torch.nn.Module,x:torch.Tensor):
        b,c,h,w = x.size()
        if isinstance(norm_layer,nn.LayerNorm):
            out = x.view(b,c,h*w).transpose(1,2)
            ##如果layernorm ，那么在h*w端做norm
            out = norm_layer(out)
            out = out.transpose(1,2).contiguous().view(b, c, h, w)
        else:
            out = norm_layer(x)
        return out

    def forward(self,x:torch.Tensor,src_mask=None):
        b,c,h,w = x.size()

        x = x.view(b,c,h*w).transpose(1,2)
        if src_mask is not None:
            src_mask = src_mask.view(b,1,h*w)
        attn_out,_=self.attention_layer(x,x,x,src_mask)

        ##add and norm
        out1 = x+attn_out
        out1 = out1.transpose(1,2).contiguous().view(b,c,h,w)
        out1 = self.norm_forward(self.attention_norm_layer,out1)

        ffn_out = self.feedforward_layer(out1)
        final_out = self.norm_forward(self.feedforward_norm_layer,out1+ffn_out)
        return final_out


class TransformerEncoderGaussianLayer2D(nn.Module):
    def __init__(self,attention_dict:dict,attention_norm_dict:dict,feedforward_dict:dict,feedforward_norm_dict:dict):
        super(TransformerEncoderGaussianLayer2D, self).__init__()
        if "type" in attention_dict.keys():
            attention_class = attention_class_dict[attention_dict.pop("type")]
        else:
            attention_class = default_attention

        if "type" in feedforward_dict.keys():
            feedforward_class = feedforward_class_dict[feedforward_dict.pop("type")]
        else:
            feedforward_class = default_feedforward

        self.attention_layer = attention_class(**attention_dict)
        self.attention_norm_layer = build_norm(attention_norm_dict)
        self.feedforward_layer = feedforward_class(**feedforward_dict)
        self.feedforward_norm_layer = build_norm(feedforward_norm_dict)

    def norm_forward(self,norm_layer:torch.nn.Module,x:torch.Tensor):
        b,c,h,w = x.size()
        if isinstance(norm_layer,nn.LayerNorm):
            out = x.view(b,c,h*w).transpose(1,2)
            ##如果layernorm ，那么在h*w端做norm
            out = norm_layer(out)
            out = out.transpose(1,2).contiguous().view(b, c, h, w)
        else:
            out = norm_layer(x)
        return out

    def forward(self,x:torch.Tensor,src_mask=None):
        ##use GaussianSelfAttention
        b,c,h,w = x.size()


        # x = x.view(b,c,h*w).transpose(1,2)
        # if src_mask is not None:
        #     src_mask = src_mask.view(b,1,h*w)
        # attn_out,_=self.attention_layer(x,x,x,src_mask)

        attn_out, _ = self.attention_layer(x)

        ##add and norm
        out1 = x+attn_out
        # out1 = out1.transpose(1,2).contiguous().view(b,c,h,w)
        out1 = self.norm_forward(self.attention_norm_layer,out1)

        ffn_out = self.feedforward_layer(out1)
        final_out = self.norm_forward(self.feedforward_norm_layer,out1+ffn_out)
        return final_out


class TransformerEncoderLayer1D(nn.Module):
    def __init__(self,attention_dict:dict,attention_norm_dict:dict,feedforward_dict:dict,feedforward_norm_dict:dict):
        super(TransformerEncoderLayer1D, self).__init__()
        if "type" in attention_dict.keys():
            attention_class = attention_class_dict[attention_dict.pop("type")]
        else:
            attention_class = default_attention

        if "type" in feedforward_dict.keys():
            feedforward_class = feedforward_class_dict[feedforward_dict.pop("type")]
        else:
            feedforward_class = default_feedforward

        self.attention_layer = attention_class(**attention_dict)
        self.attention_norm_layer = build_norm(attention_norm_dict)
        self.feedforward_layer = feedforward_class(**feedforward_dict)
        self.feedforward_norm_layer = build_norm(feedforward_norm_dict)


    def forward(self, src:torch.Tensor, src_mask=None):
        attn_out, _ = self.attention_layer(src, src, src, src_mask)
        out1 = self.attention_norm_layer(src+attn_out)

        ffn_out = self.feedforward_layer(out1)
        out2 = self.feedforward_norm_layer(out1+ffn_out)

        return out2


transformer_encoder_layer_dict = {"TransformerEncoderLayer2D":TransformerEncoderLayer2D,
                                  "TransformerEncoderLayer1D":TransformerEncoderLayer1D,
                                  "TransformerEncoderGaussianLayer2D":TransformerEncoderGaussianLayer2D}
default_encoder_layer = TransformerEncoderLayer2D

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer_dict: dict, num_layers: int, pe_dict: dict = None):
        super(TransformerEncoder, self).__init__()
        if pe_dict != None:
            if "type" in pe_dict.keys():
                pos_embedding_class = position_embedding_dict[pe_dict.pop("type")]
            else:
                pos_embedding_class = default_position_embedding
        if "type" in encoder_layer_dict.keys():
            encoder_class = transformer_encoder_layer_dict[encoder_layer_dict.pop("type")]
        else:
            encoder_class = default_encoder_layer

        self.pos_embedding_layer = pos_embedding_class(**pe_dict)

        self.layers = nn.ModuleList([encoder_class(**copy.deepcopy(encoder_layer_dict)) for _ in range(num_layers)])

    @property
    def with_pe(self):
        return hasattr(self,"pos_embedding_layer") and self.pos_embedding_layer is not None

    def forward(self,x:torch.Tensor,src_mask=None):
        if self.with_pe:
            x = self.pos_embedding_layer(x)
        for layer in self.layers:
            x = layer(x,src_mask)
        return x











