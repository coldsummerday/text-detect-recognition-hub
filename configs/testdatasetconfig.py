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
train_pipeline = [
    dict(type='ResizeRecognitionImage', img_scale=(32,100)),
    dict(type='NormalizePADToTensor', max_size=(1,32,100),PAD_type="right"),
    dict(type='Collect', keys=['img', 'label']),
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        root=data_root,
        pipeline = train_pipeline,
        charsets="ChineseCharset",
        )
)