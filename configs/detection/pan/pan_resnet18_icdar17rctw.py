batch_size = 8
dataset_type = 'IcdarDetectDataset'
train_data_root = '/data/zhb/data/commonocr/end2end/icdar2017rctw/train/'
val_data_root = '/data/zhb/data/commonocr/end2end/icdar2017rctw/val/'
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
        type="PanCPPHead",
        alpha=0.5,
        beta=0.25,
        delta_agg=0.5,
        delta_dis=3,
        ohem_ratio=3,
        reduction='mean',
        min_area = 5,
        min_score = 0.85,
        # is_output_polygon= True,
    )
)

train_cfg = dict()
test_cfg = dict()
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
    batch_size=batch_size,
    train=dict(
        type=dataset_type,
        root=train_data_root,
        pipeline = train_pipeline,
        img_channel=3,
        line_flag=True,  ##icdar17rctw format,
        gt_file_prefix = "",##icdar17rctw format,  image_id.jpg image_id.txt
        ),
    val = dict(
        type=dataset_type,
        root=val_data_root,
        pipeline = val_pipeline,
        img_channel=3,
        line_flag=True,  ##icdar17rctw format
        gt_file_prefix="",##icdar17rctw format,  image_id.jpg image_id.txt
        ),
    test=dict(
        type=dataset_type,
        root=val_data_root,
        pipeline = test_pipeline,
        line_flag=True,  ##icdar17rctw format
        gt_file_prefix="",##icdar17rctw format,  image_id.jpg image_id.txt
        )
)
# optimizer
optimizer = dict(type='Adam', lr=0.001, amsgrad=True, weight_decay=0)
dist_params = dict(backend='nccl')
train_hooks = [
    dict(
        type="CheckpointHook",
        interval=20,## 2个epoch 保存一个结果
        by_epoch=True,
        priority = 40,
    ),
    dict(
        type="SimpleTextLoggerHook",
        by_epoch=True,
        interval=200,
        priority = 100,
    ),
    dict(
        type="IterTimerHook",
        priority = 60,
    ),
    dict(
        type="WarmupAndDecayLrUpdateHook",
        base_lr=1e-4,
        warmup_lr=1e-5,
        warmup_num=5,
        lr_gamma=0.9,
        by_epoch=True,
        min_lr=1e-7,
        priority=40,
    ),
    dict(
        type="DetEvalHook",
        dataset=dict(
        type=dataset_type,
        root=val_data_root,
        pipeline=val_pipeline,
        line_flag=True,  ##icdar17rctw format
        gt_file_prefix="",##icdar17rctw format,  image_id.jpg image_id.txt
        ),
        batch_size=batch_size,
        by_epoch=True,
        interval=20,
        priority=80,
    )
    ##eval hooks

]

# runtime settings
seed = 1211
by_epoch = True
max_number = 600
# by_epoch = False
# max_number = 30000
log_level = 'INFO'
work_dir = './work_dirs/pan_restnet18_icdar17rctw/'
resume_from = None
