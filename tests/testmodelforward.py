from texthub.core.train.runner import  Runner
from texthub.modules import build_recognizer
from texthub.modules import  build_detector
from texthub.utils import Config
import torch

from texthub.datasets import build_dataset

config_file = "./configs/detection/PSE/pse_resnet50_deform_6_trainner_lr.py"
cfg = Config.fromfile(config_file)
train_dataset = build_dataset(cfg.data.train)

model = build_detector(cfg.model)

# train_data_loader = torch.utils.data.DataLoader(
#                 train_dataset, batch_size=cfg.data.batch_size,
#                 shuffle=True,
#                 pin_memory=True)

import torch
model =torch.nn.DataParallel(model).cuda()
# data = train_data_loader.__iter__().__next__()
# img = data['img']
# labels = data['label']
# b = model(img, None,return_loss=False)
# print(b)