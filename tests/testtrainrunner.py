from texthub.core.train.runner import  Runner
from texthub.modules import build_recognizer
from texthub.utils import Config
from texthub.datasets import build_dataloader
import torch

from texthub.datasets import build_dataset

config_file = "./configs/testdatasetconfig.py"
cfg = Config.fromfile(config_file)
train_dataset = build_dataset(cfg.data.train)

model = build_recognizer(cfg.model)

train_data_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=cfg.data.imgs_per_gpu,
                shuffle=True,
                num_workers=cfg.data.workers_per_gpu,
                pin_memory=True)


data = _data_loader.__iter__().__next__()
img = data['img']
labels = data['label']
b = model(img,labels, return_loss=True)
