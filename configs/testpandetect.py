# dataset settings

model = dict(
    type="PAN",
    pretrained=None,
    backbone=dict(
        type="ResNet",
        depth=18,
        arch="resnet18",
        norm="gn"
    ),
    neck=dict(
        type="FPEM_FFM",
        backbone_out_channels=[64, 128, 256, 512],
        fpem_repeat=4,
    ),
    bbox_head=dict(
        type="PanHead",
        alpha=0.5,
        beta=0.25,
        delta_agg=0.5,
        delta_dis=3,
        ohem_ratio=3,
        reduction='mean'
    )
)
train_cfg = dict()
test_cfg = dict()
dataset_type = 'IcdarDetectDataset'
data_root = '/Users/zhouhaibin/data/receipt_2nd_icdr15/'
#data_root = '/data/zhb/data/receipt/end2end/receipt_2nd_icdr15/'

train_pipeline = [
    dict(type="CheckPolys"),
    dict(type="DetectResize",img_scale=(640,640)),
    dict(type="RandomFlip",flip_ratio=0.3),
    dict(type="RandomRotate",degrees=10),
    dict(type="GenerateTrainMask"),
    dict(type="Ndarray2tensor"),
    dict(type='Collect', keys=['img', 'labels',"training_mask"]),
]

data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        root=data_root,
        pipeline = train_pipeline,
        img_channel=3,
        ),
    val=dict(
        type=dataset_type,
        root=data_root,
        pipeline = train_pipeline,
        )
)


# optimizer
optimizer = dict(type='Adam', lr=0.001, amsgrad=True, weight_decay=0)
optimizer_config = dict()

dist_params = dict(backend='nccl')
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
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 24
log_level = 'INFO'
work_dir = './work_dirs/pan/'
load_from = None
resume_from = None
workflow = [('train', 1),('val',1)]
