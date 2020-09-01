# dataset settings

model = dict(
    type="PAN",
    pretrained=None,
    backbone=dict(
        type="DetResNet",
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
data_root = '/data/zhb/data/receipt/end2end/receipt_2nd_icdr15/'
val_data_root = '/data/zhb/data/receipt/end2end/receipt_2nd_icdr15val/'
train_pipeline = [
    dict(type="CheckPolys"),
    dict(type="DetectResize",img_scale=(640,640)),
    dict(type="RandomFlip",flip_ratio=0.3),
    dict(type="RandomRotate",degrees=10),
    dict(type="GenerateTrainMask",shrink_ratio_list=[1,0.4]),
    dict(type="Ndarray2tensor"),
    dict(type='Collect', keys=['img','gt',"mask"]),
]

test_pipeline = [
    dict(type="DetectResize",img_scale=(640,640)),
    dict(type="Ndarray2tensor"),
    dict(type='Collect', keys=['img']),
]
val_pipeline = [


    dict(type="DetectResize",img_scale=(640,640)),
    dict(type="Ndarray2tensor"),
    dict(type="Gt2SameDim",max_label_num = 250),
    dict(type='Collect', keys=['img',"gt_polys"]),
]


data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        root=data_root,
        pipeline = train_pipeline,
        img_channel=3,
        line_flag=False,  ##icdar15 format
        ),
    val = dict(
        type=dataset_type,
        root=val_data_root,
        pipeline = val_pipeline,
        img_channel=3,
        line_flag=False,  ##icdar15 format
        ),
    test=dict(
        type=dataset_type,
        root=val_data_root,
        pipeline = test_pipeline,
        line_flag=False,  ##icdar15 format

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
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    by_epoch=True,
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
seed = 10
total_epochs = 24
log_level = 'INFO'
work_dir = './work_dirs/pan/'
load_from = None
resume_from = None
workflow = [('train', 1)]
