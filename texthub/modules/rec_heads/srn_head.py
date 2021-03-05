import torch
import  torch.nn as nn
import  torch.nn.functional as F
import  numpy as np
from torch.jit.annotations import Tuple, Optional
from ..builder import build_loss
from ..losses import RECSRNLoss


from ..registry import HEADS
from .labelconverter import AttnLabelConverter

@HEADS.register_module
class SRNHead(nn.Module):
    """
    unofficial code for paper:Towards Accurate Scene Text Recognition with Semantic Reasoning Networks arXiv:2003.12294v1

    """
    def __init__(self,charsets,dim_model=512,dim_hidden=512,max_len_labels=25,num_layers=2,num_heads=8,imgh=64,imgw=256,loss=None):
        super(SRNHead, self).__init__()

        hw_feature_dim = int(imgh/8)* int(imgw/8)
        self.max_len_labels = max_len_labels
        self.label_converter = AttnLabelConverter(charsets)
        self.num_classes = len(self.label_converter.character)


        self.transformer_encoder_layer = TransformerEncoder(dim_model,dim_hidden,num_layers,num_heads,dim_k=dim_model,dim_v=dim_model,conv_feat_pixel=hw_feature_dim)
        self.pvam_layer = PVAM(channels=dim_model,max_len_labels=self.max_len_labels)
        self.gsrm_layer = GSRM(dim_model=dim_model,num_classes=self.num_classes,pad_index = 0,max_len_labels=self.max_len_labels,num_heads =num_heads)
        self.vsfd_layer = VSFD(dim_hidden=dim_model,num_classes=self.num_classes)
        if loss!=None:
            self.loss_func = build_loss(loss)
        else:
            self.loss_func =RECSRNLoss()
    def init_weights(self):
        pass
    def extract_features(self,conv_features:torch.Tensor)->torch.Tensor:
        b, c, h, w = conv_features.shape
        conv_features = conv_features.view((b,c,h*w))
        conv_features=conv_features.permute(0,2,1)
        pvam_features = self.pvam_layer(conv_features)
        gsrm_features, word_out, gsrm_out = self.gsrm_layer(pvam_features)
        final_out = self.vsfd_layer(pvam_features, gsrm_features)
        return final_out,word_out,gsrm_out

    def forward_train(self,data:dict)->dict:
        conv_features = data["img"]
        text_tensor = data["label"]
        final_out, word_out, gsrm_out = self.extract_features(conv_features)
        return self.loss_func(final_out,word_out,gsrm_out,text_tensor[:,1:-1])

    def forward_test(self,data:dict)->torch.Tensor:
        conv_features = data["img"]
        final_out, word_out, gsrm_out = self.extract_features(conv_features)
        return final_out


    def forward(self,data:dict,return_loss:bool):
        if return_loss:
            return self.forward_train(data)
        else:
            return self.forward_test(data)

    def postprocess(self,preds:torch.Tensor):
        batch_size = preds.size(0)
        _, preds_index = preds.max(-1)
        preds_str = self.converter.decode(preds_index, [self.max_len_labels] * batch_size)
        scores = []
        return preds_str,scores






class PVAM(nn.Module):
    ''' Parallel Visual attention module 平行解码'''
    '''
    channels:类别数量
    '''
    def __init__(self, channels:int, max_len_labels=25):
        super(PVAM, self).__init__()
        self.max_len_labels = max_len_labels
        self.word_fc = nn.Linear(channels,channels)
        ## (vocab_size, emb_size)
        self.word_pos_embedding_layer = nn.Embedding(max_len_labels,channels)
        self.atten_fc = nn.Linear(channels,1)
        self.active = nn.Tanh()
        self.softmax_layer = nn.Softmax(dim=2)

    def forward(self, word_features:torch.Tensor):

        """
        word_features:(b,t,c)经过transformer 的特征
        """
        b,t,c = word_features.shape
        word_features = self.word_fc(word_features)


        # word_features_  = word_features.reshape([-1,1,t,c])
        word_features_ = word_features.clone().unsqueeze(1)#[b,1,t,c]
        word_features_ = word_features_.repeat(1,self.max_len_labels,1,1) #[b,max_len,t,c]
        # gsrm_word_pos = np.array(range(0, self.max_len_labels)).reshape(
        #     (self.max_len_labels, 1)).astype('int64')
        gsrm_word_pos =torch.arange(self.max_len_labels, dtype=torch.long, device=word_features.device)
        gsrm_word_pos = gsrm_word_pos.unsqueeze(0).expand(b,-1)  # (S,) -> (B, S)


        word_pos_feature = self.word_pos_embedding_layer(gsrm_word_pos) #[b,len,c]
        word_pos_feature = word_pos_feature.reshape(-1,self.max_len_labels,1,c)#[b,self.max_len,1,c]
        word_pos_feature = word_pos_feature.repeat(1,1,t,1)#[b,max_len,t,c]

        add_features = torch.tanh(word_features_ + word_pos_feature)

        #[b,max_len,t,c] -> [b,max_len,t,1]
        attn_weights = self.atten_fc(add_features)
        attn_weights = attn_weights.view(-1,self.max_len_labels,t) #[b,max_len,t]
        attention_weights = self.softmax_layer(attn_weights)
        pvam_features = torch.bmm(attention_weights,word_features)
        return pvam_features

class GSRM(nn.Module):
    """
    # global semantic reasoning module
    """
    def __init__(self,dim_model:int=512,num_classes:int=5560,pad_index = 0,max_len_labels=25,num_heads =8):
        super(GSRM, self).__init__()

        self.num_classes = num_classes
        self.pad_idx = pad_index
        self.max_len_labels = max_len_labels
        self.num_heads = num_heads
        self.word_out_fc = nn.Linear(dim_model,num_classes)
        self.word_out_embedding = nn.Embedding(num_classes,dim_model)
        self.gsrm_word_embedding_layer = nn.Linear(dim_model,num_classes)


        self.transformer_units = TransformerEncoder(dim_model=dim_model,dim_inner_hid=512,num_layers=2,num_heads=num_heads,dim_k=dim_model,dim_v=dim_model,add_bias_kv=True)

    def forward(self,pvam_features:torch.Tensor)->torch.Tensor:
        b,t,c = pvam_features.shape
        word_out = self.word_out_fc(pvam_features)
        word_out = torch.softmax(word_out,dim=-1)
        word_ids = torch.argmax(word_out, dim=-1)
        #(b,t)

        word_ids_emb = self.word_out_embedding(word_ids)
        gsrm_feature = self.transformer_units(word_ids_emb)
        gsrm_out = self.gsrm_word_embedding_layer(gsrm_feature)

        gsrm_out = torch.softmax(gsrm_out,dim=-1)

        ##word_out,gsrm_out都需要进行ce loss计算
        return gsrm_feature,word_out,gsrm_out

class VSFD(nn.Module):
    """
           Visual-Semantic Fusion Decoder Module

           args:
               pvam_features(variable):  Feature map extracted from pvam
               gsrm_features(list):  Feature map extracted from gsrm

           return: fc_out
           """

    # ===== Visual-Semantic Fusion Decoder Module =====
    def __init__(self,dim_hidden:int,num_classes=5560):
        super(VSFD, self).__init__()
        self.num_classes = num_classes

        self.combine_fc_layer = nn.Linear(dim_hidden*2,dim_hidden)
        self.final_fc_layer = nn.Linear(dim_hidden,num_classes)

    def forward(self,pvam_features:torch.Tensor,gsrm_features:torch.Tensor):
        combine_features = torch.cat([pvam_features,gsrm_features],dim=-1)
        img_combine_weights = torch.sigmoid(self.combine_fc_layer(combine_features))
        final_combine_features = img_combine_weights * pvam_features + (1-img_combine_weights)*gsrm_features
        fc_out = torch.softmax(self.final_fc_layer(final_combine_features),dim=-1)
        return fc_out


class TransformerEncoder(nn.Module):
    """
    #     d_word_vec: 位置编码，特征空间维度 = 512 dim_hidden
#     conv_feat_pixel: 位置编码的最大值(imgw/8 * imgh/8)(64/8 * 256/8 =256)
    """
    def __init__(self,dim_model,dim_inner_hid,num_layers,num_heads,dim_k,dim_v,dropout_p=0.1,conv_feat_pixel=256,add_bias_kv=True):
        super(TransformerEncoder, self).__init__()

        self.position_enc = PositionalEncoding(dim_model, n_position=conv_feat_pixel)
        self.dropout = nn.Dropout(p=dropout_p)
        self.transformer_layer_stack = nn.ModuleList([
            TransformerUnit(dim_model=dim_model,dim_inner_hid=dim_inner_hid,num_heads=num_heads,dim_k=dim_k,dim_v=dim_v,add_bias_kv=add_bias_kv)
            for _ in range(num_layers)
        ])

    def forward(self,enc_input:torch.Tensor)->torch.Tensor:
        # -- Forward
        enc_output = self.dropout(self.position_enc(enc_input))  # position embeding
        for enc_layer in self.transformer_layer_stack:
            enc_output = enc_layer(enc_output)
        return enc_output


class TransformerUnit(nn.Module):
    """
    compose(multi-head attention ,add & norm,position-wise ffn,add & norm)
    """
    def __init__(self,dim_model,dim_inner_hid,num_heads,dim_k,dim_v,dropout_p=0.1,add_bias_kv=False):
        super(TransformerUnit, self).__init__()

        """
                >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
        """
        ##最后一个维度的长度
        norm_shape =dim_model
        self.slf_attn_layer = MultiheadAttention(embed_dim=dim_model,num_heads=num_heads,kdim=dim_k,vdim=dim_v,add_bias_kv=add_bias_kv)

        self.addnorm1 = AddNorm(norm_shape, dropout_p)
        self.pos_ffn_layer = PositionwiseFeedForward(dim_model, dim_inner_hid, dropout=dropout_p)
        self.addnorm2 = AddNorm(norm_shape, dropout_p)

    def forward(self, enc_input:torch.Tensor, slf_attn_mask=None):
        attn_output, attn_output_weights  = self.slf_attn_layer(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        Y = self.addnorm1(enc_input,attn_output)
        enc_output = self.addnorm2(Y,self.pos_ffn_layer(Y))
        return enc_output







#[max_len=256,d_model=512]
#[src_max_len, src_emb_dim]
# class SRNEncoder(nn.Module):
#     """
#     d_word_vec: 位置编码，特征空间维度
#     conv_feat_pixel: 位置编码的最大值(imgw/8 * imgh/8)(64/8 * 256/8 =256)
#     """
#     def __init__(self,d_word_vec=512,conv_feat_pixel=256,dropout_p=0.1):
#         super(SRNEncoder, self).__init__()
#         self.position_embedding_layer = PositionalEncoding(d_word_vec, n_position=conv_feat_pixel)
#         self.dropout_layer = nn.Dropout(p=dropout_p)
#
#     def forward(self,conv_features:torch.Tensor):
#         b,c,h,w = conv_features.shape
#         feature_dim = h * w
#
#         # encoder_word_pos = np.array(range(0, feature_dim)).reshape(
#         #     (feature_dim, 1)).astype('int64')
#         encoder_word_pos = torch.arange(feature_dim, dtype=torch.long, device=conv_features.device)
#         encoder_word_pos = encoder_word_pos.unsqueeze(0).expand(b, -1)  # (S,) -> (B, w*h)
#         pass










# class TransformerEncoder(nn.Module):
#     def __init__(self):
#         super(TransformerEncoder, self).__init__()
#         self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
#


"""
transformer 
"""
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_in:int,d_hid:int,dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_in, d_hid)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hid, d_in)
        self.norm1 = nn.LayerNorm(d_in)
        self.dropout1 = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        src = x + self.dropout1(src2)
        src = self.norm1(src)
        return src



class PositionwiseFeedForwardCNN(nn.Module):
    ''' A two-feed-forward-layer module '''

    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForwardCNN, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class AddNorm(nn.Module):
    def __init__(self, normalized_shape:int, dropout_p=0.1, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout_p)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim,bias=True)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:

            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()