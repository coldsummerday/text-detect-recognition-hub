
batch_size = 8
model = dict(
    type="DBDetector",
    pretrained=None,
    backbone=dict(
        type="DetResNet",
        depth=34,
        arch="resnet34",
        norm="bn",
        stage_with_dcn=[False, True, True, True],
        dcn_config=dict(
            modulated=True,
            deformable_groups=1
        )
    ),
    neck=dict(
        type="SegDBNeck",
        in_channels=[64, 128, 256, 512],
        inner_channels = 256,
    ),
    det_head=dict(
        type="DBHead",
        inner_channels=256,
        neck_out_channels=256 // 4,
        k=50,
        thresh=0.2,
        score_thresh=0.5,
        max_candidates=1000,
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
    dict(type="GenerateTrainMask",shrink_ratio_list=[0.4]),
    dict(type="MakeBorderMap"),
    dict(type="Ndarray2tensor"),
    dict(type='Collect', keys=['img','gt',"mask","thresh_map","thresh_mask"]),
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
        root=data_root,
        pipeline = train_pipeline,
        img_channel=3,
        line_flag = False
        ),
    val = dict(
        type=dataset_type,
        root=val_data_root,
        pipeline = val_pipeline,
        img_channel=3,
        line_flag = False
        ),
    test=dict(
        type=dataset_type,
        root=val_data_root,
        pipeline = test_pipeline,
        img_channel=3,
        line_flag = False
        )
)

# optimizer
optimizer = dict(type='Adam', lr=1e-4, weight_decay=0.0001)
dist_params = dict(backend='nccl')
train_hooks = [
    dict(
        type="CheckpointHook",
        interval=5,## 5个epoch 保存一个结果
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
        line_flag=False,  ##icdar15 format
        ),
        batch_size=batch_size,
        by_epoch=True,
        interval=5,
        priority=80,
    )
    ##eval hooks

]

# runtime settings
seed = 10
by_epoch = True
max_number = 150
# by_epoch = False
# max_number = 30000
log_level = 'INFO'
work_dir = './work_dirs/db/db_resnet34_deform_adam/'
resume_from = None
