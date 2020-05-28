from texthub.core.train.runner import  Runner
from texthub.modules import build_detector
from texthub.utils import Config
import torch
from texthub.utils.dist_utils import init_dist
from texthub.datasets import build_dataset
from torch.nn.parallel import DistributedDataParallel,DataParallel

config_file = "./configs/testpandetect.py"
cfg = Config.fromfile(config_file)
train_dataset = build_dataset(cfg.data.test)
init_dist("pytorch", **cfg.dist_params)
model = build_detector(cfg.model)

train_data_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=cfg.data.imgs_per_gpu,
                num_workers=cfg.data.workers_per_gpu,
                pin_memory=True)
model = DistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)

data = train_data_loader.__iter__().__next__()
img = data['img']
labels = data['label']
b = model(img, None,return_loss=False)
print(b)
