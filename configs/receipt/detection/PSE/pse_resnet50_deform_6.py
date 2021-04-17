result_num = 6
img_size = 640
batch_size = 8
model = dict(
    type="PSEDetector",
    pretrained=None,
    backbone=dict(
        type="DetResNet",
        depth=50,
        arch="resnet50",
        norm="bn",
        stage_with_dcn=[False, True, True, True],
        dcn_config=dict(
            modulated=True,
            deformable_groups=1
        )
    ),
    neck=dict(
        type="PseFPN",
        input_channels=[256, 512, 1024, 2048],
        conv_out = 256,
    ),
    det_head=dict(
        type="PseHead",
        conv_out=256,
        result_num=result_num,
        ori_w=img_size,
        ori_h=img_size,
        Lambda=0.7, ##loss_kernel 占的比重
        OHEM_ratio=3,
        pred_threshold=0.7311,
        pred_score=0.93,
    )
)
train_cfg = dict()
test_cfg = dict()
dataset_type = 'IcdarDetectDataset'
data_root = '/data/zhb/data/receipt/end2end/receipt_2nd_icdr15/'
val_data_root = '/data/zhb/data/receipt/end2end/receipt_2nd_icdr15val/'





train_pipeline = [
    dict(type="CheckPolys"),
    dict(type="RandomScalePSE",scales=[0.5,1.0,2.0,3.0],input_size=img_size),
    dict(type="RandomFlip",flip_ratio=0.5),
    dict(type="RandomRotate",degrees=10),
    dict(type="GenerateTrainMaskPSE",result_num=result_num,m=0.5),
    dict(type="RandomCropPSE",size=640),
    dict(type="Ndarray2tensor"),
    dict(type='Collect', keys=['img','gt',"mask"]),
]

test_pipeline = [
    dict(type="DetectResize",img_scale=(img_size,img_size)),
    dict(type="Ndarray2tensor"),
    dict(type='Collect', keys=['img']),
]
val_pipeline = [
    dict(type="DetectResize",img_scale=(img_size,img_size)),
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
optimizer = dict(type='Adam', lr=1e-4, weight_decay=5e-4)
dist_params = dict(backend='nccl')
train_hooks = [
    dict(
        type="CheckpointHook",
        interval=5,## 2个epoch 保存一个结果
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
seed = 1211
by_epoch = True
max_number = 120
# by_epoch = False
# max_number = 30000
log_level = 'INFO'
work_dir = './work_dirs/pse_resnet50_deform_6/'
resume_from = None
