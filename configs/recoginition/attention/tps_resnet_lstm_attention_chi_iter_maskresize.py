# dataset settings
dataset_type = 'HierarchicalLmdbDataset'
mj_data_root = "/data/zhb/data/commonocr/data_lmdb_release/training/MJ/MJ_train/"
st_data_root = "/data/zhb/data/commonocr/data_lmdb_release/training/ST/"
train_data_roots = [mj_data_root,st_data_root]
val_data_root = '/data/zhb/data/commonocr/data_lmdb_release/validation/'

##TODO:做test
val_dataset_type = ""

batch_size=256
charsets = "EnglishPrintableCharset"
max_len_labels = 25
model = dict(
    type="FourStageModel",
    pretrained=None,
    transformation = dict(
        type="TPSSpatialTransformerNetwork",
        F=20,#'number of fiducial points of TPS-STN'
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
        type="AttentionHead",
        input_size=256,
        hidden_size=256,
        charsets=charsets,
    )
)
cudnn_benchmark = True
train_cfg = dict()
test_cfg = dict()

train_pipeline = [
    dict(type='ResizeRecognitionImageCV2', img_scale=(32,100)),
    dict(type='RecognitionImageCV2Tensor'),
    dict(type="AttentionLabelEncode",charsets=charsets,batch_max_length=max_len_labels),
    dict(type='Collect', keys=['img', 'label',"ori_label"]),
]
val_pipeline = [
    dict(type='ResizeRecognitionImageCV2', img_scale=(32, 100)),
    dict(type='RecognitionImageCV2Tensor'),
    dict(type='Collect', keys=['img', 'label']),
]
test_pipeline = [
    dict(type='ResizeRecognitionImageCV2', img_scale=(32, 100)),
    dict(type='RecognitionImageCV2Tensor'),
    dict(type='Collect', keys=['img']),
]

data = dict(
    batch_size=batch_size,
    train=dict(
        type=dataset_type,
        root=train_data_roots,
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
dist_params = dict(backend='nccl')
# by_epoch = True
# max_number = 500
by_epoch = False
max_number = 300000
train_hooks = [
    dict(
        type="CheckpointHook",
        interval=50000,## 5个epoch 保存一个结果
        by_epoch=by_epoch,
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
        root=val_data_root,
        pipeline = val_pipeline,
        charsets=charsets,
        ),
        batch_size=batch_size,
        by_epoch=by_epoch,
        interval=2000,
        priority=80,
    )
    ##eval hooks

]

# runtime settings
seed = 10

log_level = 'INFO'
work_dir = './work_dirs/common/attention_mj_st/'
resume_from = None
