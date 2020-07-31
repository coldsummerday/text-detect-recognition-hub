model = dict(
    type="DBDetector",
    pretrained=None,
    backbone=dict(
        type="DetResNet",
        depth=34,
        arch="resnet34",
        norm="gn",
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
        thresh=0.3,
        score_thresh=0.7,
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
    dict(type="NormalizeImage"),
    dict(type='Collect', keys=['img','gt',"mask","thresh_map","thresh_mask"]),
]
test_pipeline = [
    dict(type="DetectResize",img_scale=(640,640)),
    dict(type="NormalizeImage"),
    dict(type='Collect', keys=['img']),
]
val_pipeline = [
    dict(type="DetectResize",img_scale=(640,640)),
    dict(type="NormalizeImage"),
    dict(type="Gt2SameDim",max_label_num = 250),
    dict(type='Collect', keys=['img',"gt_polys"]),
]

data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=4,
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
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict()

dist_params = dict(backend='nccl')
# learning policy
optimizer_config = dict()
##不使用lr减少
lr_config = dict(
    policy="ExpIterdecay",
    interval=5000,
    power_decay = 0.9,
    min_lr=0.005,
)
checkpoint_config = dict(interval=30000) ##save_mode true->epoch, false->iter
dist_params = dict(backend='nccl')
# yapf:disable
log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

seed = 1211
total_iters = 300000
log_level = 'INFO'
work_dir = './work_dirs/db_resnet34_deform/'
load_from = None
resume_from = None
workflow = [('train', 1)]
