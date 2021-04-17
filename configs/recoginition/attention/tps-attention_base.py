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

# 192 - 8G,512-19G
batch_size=512
charsets = "EnglishNoSensitiveCharset"
max_len_labels = 25
img_h = 32
img_w = 100

model = dict(
    type="FourStageModel",
    pretrained=None,
    transformation = dict(
        type="TPSSpatialTransformerNetwork",
        F=20,#'number of fiducial points of TPS-STN'
        I_size=(img_h, img_w),
        I_r_size=(img_h, img_w),
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
        type="AttentionHead",
        input_size=256,
        hidden_size=256,
        batch_max_length=max_len_labels,
        charsets=charsets,
    )
)
cudnn_benchmark = True
train_cfg = dict()
test_cfg = dict()

train_pipeline = [
    dict(type='ResizeRecognitionFixWhImage', img_scale=(32,100)),
    dict(type='ToTensorRecognition'),
    dict(type="NormalizeRecognition",mean=[0.5], std=[0.5]),
    dict(type="AttentionLabelEncode",charsets=charsets,batch_max_length=max_len_labels),
    dict(type='Collect', keys=['img', 'label',"ori_label"]),
]
val_pipeline = [
    dict(type='ResizeRecognitionFixWhImage', img_scale=(32, 100)),
    dict(type='ToTensorRecognition'),
    dict(type="NormalizeRecognition", mean=[0.5], std=[0.5]),
    dict(type='Collect', keys=['img', 'label']),
]
test_pipeline = [
    dict(type='ResizeRecognitionFixWhImage', img_scale=(32, 100)),
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
by_epoch = False
train_hooks = [
    dict(
        type="CheckpointHook",
        interval=40000,
        by_epoch=by_epoch,
        priority = 40,
    ),
    dict(
        type="SimpleTextLoggerHook",
        by_epoch=by_epoch,
        interval=400,
        priority = 100,
    ),
    dict(
        type="MultiStepLrUpdateHook",
        priority=80,
        base_lr=1,
        by_epoch = by_epoch,
        iters_list=[240000],
        lr_gamma=0.1
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
        by_epoch=by_epoch,
        interval=2000,
        priority=80,
    )
    ##eval hooks
]

# runtime settings
seed = 1211

max_number = 420000
log_level = 'INFO'
work_dir = './work_dirs/commonocr/attention/tps-attention/'
resume_from = None
