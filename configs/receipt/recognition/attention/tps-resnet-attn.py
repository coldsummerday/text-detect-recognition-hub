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

batch_size=32
charsets = "ChineseCharset"
max_len_labels = 50
img_h = 48
img_w = 160

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
    dict(type='ResizeRecognitionImageCV2', img_scale=(img_h,img_w)),
    dict(type='RecognitionImageCV2Tensor'),
    dict(type="AttentionLabelEncode",charsets=charsets,batch_max_length=max_len_labels),
    dict(type='Collect', keys=['img', 'label',"ori_label"]),
]
val_pipeline = [
    dict(type='ResizeRecognitionImageCV2', img_scale=(img_h,img_w)),
    dict(type='RecognitionImageCV2Tensor'),
    dict(type='Collect', keys=['img', 'label']),
]
test_pipeline = [
    dict(type='ResizeRecognitionImageCV2', img_scale=(img_h, img_w)),
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
        interval=20,## 20个epoch 保存一个结果
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
        interval=20,
        priority=80,
    )
    ##eval hooks

]

# runtime settings
seed = 1211
max_number = 100
log_level = 'INFO'
work_dir = './work_dirs/receipt/attention/tps-resnet-attention/'
resume_from = None
