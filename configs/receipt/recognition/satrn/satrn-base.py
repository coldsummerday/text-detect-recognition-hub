# dataset settings
dataset_type = 'ConcateLmdbDataset'
base_dataset_type = "LmdbWorkersDataset"
train_dataset_type = "ConcateLmdbDataset"

train_data_root = [

    "/data/zhb/data/receipt/end2end/receipt_23nd_icdar/lmdb/train/",
    "/data/zhb/data/receipt/end2end/receipt_23nd0126_icdar/lmdb/"
]
val_data_root = [
    '/data/zhb/data/receipt/end2end/receipt_23nd0126_val/lmdb/',
]

batch_size = 64
charsets = "ChineseCharset"
# max_len_labels = 50
# img_h = 48
# img_w = 160
max_len_labels = 35
img_h = 32
img_w = 100

in_channels = 1
hidden_dim = 256
num_encoder = 9
num_decoder = 3
num_head = 8
dropout = 0.1

model = dict(
    type="Satrn",
    pretrained=None,
    backbone=dict(
        type="TransformerResNet",
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        encoder_dict=dict(
            num_layers=num_encoder,
            pe_dict=dict(
                type="Adaptive2DPositionEncoder",
                in_channels=hidden_dim,
                max_h=100,
                max_w=100,
                dropout=dropout,
            ),
            encoder_layer_dict=dict(
                type="TransformerEncoderLayer2D",
                attention_dict=dict(
                    type='MultiHeadAttention',
                    in_channels=hidden_dim,
                    k_channels=hidden_dim // num_head,
                    v_channels=hidden_dim // num_head,
                    n_head=num_head,
                    dropout=dropout,
                ),
                attention_norm_dict=dict(
                    type="LN",
                    dim=hidden_dim
                ),
                feedforward_dict=dict(
                    type="ConvFeedforward",
                    hidden_dim=hidden_dim,
                    dropout=dropout,

                ),
                feedforward_norm_dict=dict(
                    type="LN",
                    dim=hidden_dim
                ),
            ),
        )
    ),
    label_head=dict(
        type="TransfomerHead",
        max_len_labels=max_len_labels,
        hidden_dim=hidden_dim,
        ignore_index=0,
        charsets=charsets,
        decoder_dict=dict(
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
                    k_channels=hidden_dim // num_head,
                    v_channels=hidden_dim // num_head,
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

    )
)
cudnn_benchmark = True
train_cfg = dict()
test_cfg = dict()

train_pipeline = [
    dict(type='ResizeRecognitionImageCV2', img_scale=(img_h,img_w),img_channel=1),
    dict(type='RecognitionImageCV2Tensor'),
    dict(type="AttentionLabelEncode",charsets=charsets,batch_max_length=max_len_labels),
    dict(type='Collect', keys=['img', 'label',"ori_label"]),
]
val_pipeline = [
    dict(type='ResizeRecognitionImageCV2', img_scale=(img_h,img_w),img_channel=1),
    dict(type='RecognitionImageCV2Tensor'),
    dict(type='Collect', keys=['img', 'label']),
]
test_pipeline = [
    dict(type='ResizeRecognitionImageCV2', img_scale=(img_h, img_w),img_channel=1),
    dict(type='RecognitionImageCV2Tensor'),
    dict(type='Collect', keys=['img']),
]

data = dict(
    batch_size=batch_size,
    train=dict(
        type=train_dataset_type,
        base_dataset_type=base_dataset_type,
        root=train_data_root,
        pipeline = train_pipeline,
        charsets=charsets,
        batch_max_length = max_len_labels,
        ),
    val=dict(
        type=dataset_type,
        root=val_data_root,
        base_dataset_type=base_dataset_type,
        pipeline = val_pipeline,
        charsets=charsets,
        batch_max_length = max_len_labels,
        ),
    test=dict(
        type=dataset_type,
        root=val_data_root,
        pipeline = val_pipeline,
        charsets=charsets,
        )
)

# optimizer
optimizer = dict(type='Adadelta', lr=1, rho=0.95, eps=1e-8)
dist_params = dict(backend='nccl')
by_epoch = True
train_hooks = [
    dict(
        type="CheckpointHook",
        interval=10,  ## 10个epoch 保存一个结果
        by_epoch=by_epoch,
        priority=40,
    ),
    dict(
        type="SimpleTextLoggerHook",
        by_epoch=by_epoch,
        interval=400,
        priority=100,
    ),
    dict(
        type="RNNClipGradHook",
        priority=90,
        grad_clip=5,  # Norm cutoff to prevent explosion of gradients'
    ),
    dict(
        type="IterTimerHook",
        priority=60,
    ),
    dict(
        type="RecoEvalHook",
        dataset=dict(
            type=dataset_type,
            base_dataset_type=base_dataset_type,
            root=val_data_root,
            pipeline=val_pipeline,
            charsets=charsets,
            batch_max_length=max_len_labels,
        ),
        batch_size=batch_size,
        by_epoch=by_epoch,
        interval=20,
        priority=80,
    )
    ##eval hooks

]

# runtime settings
seed = 1211
by_epoch = True
max_number = 200

log_level = 'INFO'
work_dir = './work_dirs/receipt/satrn/satrn-base/'
resume_from = None
