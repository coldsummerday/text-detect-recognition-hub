# dataset settings

model = dict(
    type="Seq2SeqAttention",
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

    label_head =dict(
        type="Seq2SeqAttnHead",
        input_size=512,
        hidden_size=256,
        charsets="ChineseCharset",
        batch_max_length=25
    ),
    feature_wh = 24,
    batch_max_length = 25,
)
cudnn_benchmark = True
train_cfg = dict()
test_cfg = dict()
dataset_type = 'LmdbDataset'
data_root = '/data/zhb/data/receipt/TextRecognition/3rd_lmdb_recognition_benchmark_data/train_lmdb_benchmark/'
val_data_root = '/data/zhb/data/receipt/TextRecognition/3rd_lmdb_recognition_benchmark_data/for_valid/test_lmdb_benchmark/'
#val_data_root = "/Users/zhouhaibin/data/for_valid/test_lmdb_benchmark/"
train_pipeline = [
    dict(type='ResizeRecognitionImage', img_scale=(32,100)),
    dict(type='NormalizePADToTensor', max_size=(1,32,100),PAD_type="right"),
    dict(type="AttentionLabelEncode",charsets="ChineseCharset",batch_max_length=25),
    dict(type='Collect', keys=['img', 'label',"ori_label"]),
]
val_pipeline = [
    dict(type='ResizeRecognitionImage', img_scale=(32,100)),
    dict(type='NormalizePADToTensor', max_size=(1,32,100),PAD_type="right"),
    dict(type='Collect', keys=['img', 'label']),
]
test_pipeline = [
    dict(type='ResizeRecognitionImage', img_scale=(32, 100)),
    dict(type='NormalizePADToTensor', max_size=(1, 32, 100), PAD_type="right"),
    dict(type='Collect', keys=['img',"label"]),
]

##128每张显存 2613MiB,256:5207MiB
data = dict(
    imgs_per_gpu=256,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        root=data_root,
        pipeline = train_pipeline,
        charsets="ChineseCharset",
        ),
    val=dict(
        type=dataset_type,
        root=val_data_root,
        pipeline = val_pipeline,
        charsets="ChineseCharset",
        ),
    test=dict(
        type=dataset_type,
        root=val_data_root,
        pipeline = val_pipeline,
        charsets="ChineseCharset",
        )
)

# optimizer
optimizer = dict(type='Adadelta', lr=1, rho=0.95, eps=1e-8)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
checkpoint_config = dict(interval=50000,save_mode=False) ##save_mode true->epoch, false->iter
dist_params = dict(backend='nccl')
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_iters = 300000
log_level = 'INFO'
work_dir = './work_dirs/tps_vgg_seq2seq_attention/'
load_from = None
resume_from = None
workflow = [('train', 1)]
