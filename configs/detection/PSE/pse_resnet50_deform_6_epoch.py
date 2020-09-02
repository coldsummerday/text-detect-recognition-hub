result_num = 6
img_size = 640
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
optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()

dist_params = dict(backend='nccl')
# learning policy
optimizer_config = dict()
##不使用lr减少
lr_config = dict(
    policy="ExpIterdecay",
    interval=200,
    power_decay = 0.9,
    min_lr=1e-7,
)

dist_params = dict(backend='nccl')
# learning policy
checkpoint_config = dict(interval=50,by_epoch=True)
# yapf:disable
log_config = dict(
    interval=200,
    by_epoch=True,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
seed = 10
total_epochs = 600
log_level = 'INFO'
work_dir = './work_dirs/pse/pse_resnet50_deform_6_epoch/'
load_from = None
resume_from = None
workflow = [('train', 1)]



