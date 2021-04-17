# dataset settings
dataset_type = 'ConcateLmdbDataset'
base_dataset_type = "LmdbPILWorkersDataset" ##公榜数据用PIL数据读取方式
train_dataset_type = "ConcateLmdbDataset"

train_data_root = [
    "/data/zhb/data/commonocr/data_lmdb_release/training/ST/",
    "/data/zhb/data/commonocr/data_lmdb_release/training/MJ/MJ_train/",
    "/data/zhb/data/commonocr/data_lmdb_release/training/MJ/MJ_test/",
    "/data/zhb/data/commonocr/data_lmdb_release/training/MJ/MJ_valid/",
]
val_data_root = [
    "/data/zhb/data/commonocr/data_lmdb_release/validation/"
]

# 128-14G
batch_size=196
charsets = "EnglishNoSensitiveCharset"
max_len_labels = 25
img_h = 32
img_w = 100
sensitive = True

in_channels = 1
hidden_dim = 256
num_encoder = 9
num_decoder = 3
num_head = 9
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
                type="PositionEncoder1D",
                in_channels=hidden_dim,
                max_len=200,
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
                    n_head=num_head,
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
                    n_head=num_head,
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
    dict(type='ResizeRecognitionFixWhImage', img_scale=(img_h,img_w)),
    dict(type='ToTensorRecognition'),
    dict(type="NormalizeRecognition",mean=[0.5], std=[0.5]),
    dict(type="AttentionLabelEncode",charsets=charsets,batch_max_length=max_len_labels),
    dict(type='Collect', keys=['img', 'label',"ori_label"]),
]
val_pipeline = [
    dict(type='ResizeRecognitionFixWhImage', img_scale=(img_h,img_w)),
    dict(type='ToTensorRecognition'),
    dict(type="NormalizeRecognition", mean=[0.5], std=[0.5]),
    dict(type='Collect', keys=['img', 'label']),
]
test_pipeline = [
    dict(type='ResizeRecognitionFixWhImage', img_scale=(img_h,img_w)),
    dict(type='ToTensorRecognition'),
    dict(type="NormalizeRecognition", mean=[0.5], std=[0.5]),
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
train_hooks = [
    dict(
        type="CheckpointHook",
        interval=2,
        by_epoch=True,
        priority = 40,
    ),
    dict(
        type="SimpleTextLoggerHook",
        by_epoch=True,
        interval=400,
        priority = 100,
    ),
    dict(
        type="RNNClipGradHook",
        priority=90,
        grad_clip = 5,#Norm cutoff to prevent explosion of gradients'
    ),
    dict(
        type="IterTimerHook",
        priority = 60,
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
        by_epoch=True,
        interval=2,
        priority=80,
    )
    ##eval hooks

]

# runtime settings
seed = 1211
by_epoch = True
max_number = 100
log_level = 'INFO'
work_dir = './work_dirs/commonocr/transformer/transformer-base/'
resume_from = None
