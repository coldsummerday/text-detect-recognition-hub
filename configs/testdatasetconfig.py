# dataset settings
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