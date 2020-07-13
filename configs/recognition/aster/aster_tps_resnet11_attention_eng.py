# dataset settings
dataset_type = 'LmdbDataset'
data_root = '/data/zhb/data/receipt/TextRecognition/data_lmdb_release/training/ST/'
val_data_root = '/data/zhb/data/receipt/TextRecognition/data_lmdb_release/validation/'

charsets = "EnglishPrintableCharset"
max_len_labels = 25
model = dict(
    type="AsterRecognizer",
    pretrained=None,
    transformation = dict(
        type="SPN",
        K=20,#'number of fiducial points of TPS-STN'
        I_size=(32, 100),
        I_r_size=(32, 100),
        I_channel_num=1
    ),
    backbone=dict(
        type="AsterResNet",
        input_channel=1,
        output_channel=512
    ),
    sequence = dict(
        type="DoubleBidirectionalLSTM",
        input_size=512,
        hidden_size=256,
    ),
    label_head =dict(
        type="AsterAttentionRecognitionHead",
        input_dim=256,
        hidden_dim=256,
        attention_dim=256,
        beam_width=2,
        charsets=charsets,
        max_len_labels=max_len_labels
    ),
)
cudnn_benchmark = True
train_cfg = dict()
test_cfg = dict()

train_pipeline = [
    dict(type='ResizeRecognitionImage', img_scale=(32,100)),
    dict(type='NormalizePADToTensor', max_size=(1,32,100),PAD_type="right"),
    dict(type="AttentionLabelEncode",charsets=charsets,batch_max_length=max_len_labels),
    dict(type='Collect', keys=['img', 'label',"ori_label"]),
]
val_pipeline = [
    dict(type='ResizeRecognitionImage', img_scale=(32,100)),
    dict(type='NormalizePADToTensor', max_size=(1,32,100),PAD_type="right"),
    dict(type='Collect', keys=['img', 'label']),
]
test_pipeline = [
    dict(type='ResizeRecognitionImage', img_scale=(32, 100)),
    dict(type='NormalizePADToTensor', max_size=(1, 32, 100), PAD_type="right"),
    dict(type='Collect', keys=['img']),
]

##128每张显存 3843MiB,256:5207MiB
data = dict(
    imgs_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        root=data_root,
        pipeline = train_pipeline,
        charsets=charsets,
        ),
    val=dict(
        type=dataset_type,
        root=val_data_root,
        pipeline = val_pipeline,
        charsets=charsets,
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
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
checkpoint_config = dict(interval=10,save_mode=True) ##save_mode true->epoch, false->iter
dist_params = dict(backend='nccl')
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 70
log_level = 'INFO'
work_dir = './work_dirs/aster_tps_resnet_attion_eng/'
load_from = None
resume_from = None
workflow = [('train', 1)]