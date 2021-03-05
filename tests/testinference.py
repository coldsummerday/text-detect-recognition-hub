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


backbone_cfg = dict(
    type="ResnetDilated",
orig_resnet = dict(
    depth=18,
        arch="resnet18",
        norm="gn"
)
)

import  torch
from texthub.modules.losses.rec_ctc_loss2d import CTCLoss2D
ctc_loss = CTCLoss2D()
N, H, T, C = 16, 8, 32, 20
mask = torch.randn(T, H, N).log_softmax(1).detach().requires_grad_()
classify = torch.randn(T, H, N, C).log_softmax(3).detach().requires_grad_()
targets = torch.randint(1, C, (N, C), dtype=torch.long)
input_lengths = torch.full((N,), T, dtype=torch.long)
target_lengths = torch.randint(10, 31, (N,), dtype=torch.long)
loss = ctc_loss(mask, classify, targets, input_lengths, target_lengths)
loss.backward()

# from texthub.modules.backbones.resnet_dilated import ResnetDilated
# a = ResnetDilated(orig_resnet = dict(
#     type="DetResNet",
#     depth=50,
#         arch="resnet50",
#         norm="gn"
# ))
# input_tensor = torch.ones((3,3,64,256))
# b = a(input_tensor)
# from texthub.modules.necks import PPMDeepsup
# ppm = PPMDeepsup()
# c = ppm(b)
#
from torchvision.models import resnet18
model = resnet18()
from  torch.optim.adam import Adam
a = Adam()