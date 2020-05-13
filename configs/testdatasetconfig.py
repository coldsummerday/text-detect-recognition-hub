# dataset settings

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
        type="CRNNVGG",
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
        charsets="ChineseCharset"
    )
)
train_cfg = dict()
test_cfg = dict()
dataset_type = 'LmdbDataset'
data_root = '/Users/zhouhaibin/data/for_valid/test_lmdb_benchmark/'
#data_root = '/data/zhb/data/receipt/TextRecognition/3rd_lmdb_recognition_benchmark_data/for_valid/test_lmdb_benchmark/'

train_pipeline = [
    dict(type='ResizeRecognitionImage', img_scale=(32,100)),
    dict(type='NormalizePADToTensor', max_size=(1,32,100),PAD_type="right"),
    dict(type="AttentionLabelEncode",charsets="ChineseCharset",batch_max_length=25),
    dict(type='Collect', keys=['img', 'label',"ori_label"]),
]
test_pipeline = [
    dict(type='ResizeRecognitionImage', img_scale=(32, 100)),
    dict(type='NormalizePADToTensor', max_size=(1, 32, 100), PAD_type="right"),
    dict(type='Collect', keys=['img']),
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        root=data_root,
        pipeline = train_pipeline,
        charsets="ChineseCharset",
        ),
    val=dict(
        type=dataset_type,
        root=data_root,
        pipeline = train_pipeline,
        charsets="ChineseCharset",
        )
)


# optimizer
optimizer = dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 24
log_level = 'INFO'
work_dir = './work_dirs/fourstage/'

workflow = [('train', 1),('val',1)]
