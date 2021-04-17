

from texthub.modules.backbones import TransformerResNet
from texthub.modules.builder import build_backbone

in_channels = 1
hidden_dim = 256
num_encoder = 2
num_decoder = 3
num_head=8
dropout = 0.1
model_dict = dict(
    type="FourStageModel",
    pretrained=None,
    backbone=dict(
        type="TransformerResNet",
        in_channels=1,
        hidden_dim=256,
        encoder_dict=dict(
            num_layers=num_encoder,
            pe_dict=dict(
                type="Adaptive2DPositionEncoder",
                in_channels=hidden_dim,
                max_h=100,
                max_w=100,
                drop_out=dropout,
            ),
            encoder_layer_dict=dict(
                type="TransformerEncoderGaussianLayer2D",
                attention_dict=dict(
                    type='GaussianSelfAttention',
                    num_heads=num_head,
                    hidden_size=hidden_dim,
                    init_sigma_std=0.01,
                    init_mu_std=2.0,
                    attention_isotropic_gaussian=False,
                    max_width_height=100,
                    dropout=0.1,
                ),
                attention_norm_dict=dict(
                    type="BN",
                    dim=hidden_dim
                ),
                feedforward_dict=dict(
                    type="ConvFeedforward",
                    hidden_dim=hidden_dim,
                    dropout=dropout,

                ),
                feedforward_norm_dict=dict(
                    type="BN",
                    dim=hidden_dim
                ),
            ),
        )
    ),
)
decoder_layer_dict = dict(
        pe_dict=dict(
                    type='PositionEncoder1D',
                    in_channels=hidden_dim,
                    max_len=100,
                    dropout=dropout,
                ),
        num_layers=num_decoder,
        decoder_layer_dict=dict(
            type="TransformerDecoderLayer1D",
            self_attention_dict=dict(
                type="MultiHeadAttention",
                in_channels=hidden_dim,
                k_channels=hidden_dim//num_head,
                v_channels=hidden_dim//num_head,
                n_head=8,
                dropout=dropout
            ),
            self_attention_norm_dict=dict(
                    type="LN",
                    dim=hidden_dim
            ),
            attention_dict=dict(
                type="MultiHeadAttention",
                in_channels=hidden_dim,
                k_channels=hidden_dim // num_head,
                v_channels=hidden_dim // num_head,
                n_head=8,
                dropout=dropout
            ),
            attention_norm_dict=dict(
                    type="LN",
                    dim=hidden_dim
            ),
            feedforward_dict=dict(
                type="FCFeedforward",
                hidden_dim=hidden_dim,
                dropout=dropout,
            ),
            feedforward_norm_dict=dict(
                type="LN",
                dim=hidden_dim
            )
        )
)


import torch
model = build_backbone(model_dict['backbone'])
a = torch.ones((3,1,32,100))
x = model(a)
x.shape





