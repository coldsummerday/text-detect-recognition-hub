from texthub.utils import Config
import  time
import copy
from texthub.apis import  train_recoginizer
from texthub.datasets import build_dataset
from texthub.modules import build_detector
from texthub.utils import  get_root_logger
import os.path as osp
import os
config_file = "./configs/testpandetect.py"
cfg = Config.fromfile(config_file)
cfg.gpus = 1
cfg.resume_from=None
cfg.load_from = None
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
os.makedirs(cfg.work_dir,exist_ok=True)
log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
datasets = [build_dataset(cfg.data.train)]

if len(cfg.workflow) == 2:
    val_dataset = copy.deepcopy(cfg.data.val)
    val_dataset.pipeline = cfg.data.train.pipeline
    datasets.append(build_dataset(val_dataset))


train_recoginizer(
        model,
        datasets,
        cfg,
        validate=True,
        timestamp=timestamp,
        meta=None
)