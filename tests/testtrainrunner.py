from texthub.core.train.runner import  Runner
from texthub.modules import build_recognizer
from texthub.utils import Config
import torch
from texthub.utils.dist_utils import init_dist

from texthub.datasets import build_dataset


config_file = "./configs/testdatasetconfig.py"
cfg = Config.fromfile(config_file)
init_dist("")
train_dataset = build_dataset(cfg.data.train)

model = build_recognizer(cfg.model)

train_data_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=cfg.data.imgs_per_gpu,
                num_workers=cfg.data.workers_per_gpu,
                pin_memory=True)

data = train_data_loader.__iter__().__next__()
