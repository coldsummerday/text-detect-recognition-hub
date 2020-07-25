import os.path as osp
import os
import sys
this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(osp.join(this_path,'../'))
from texthub.modules import build_head,build_backbone,build_neck


import  torchvision.transforms as transforms
from PIL import  Image
backbone=dict(
        type="DetResNet",
        depth=18,
        arch="resnet18",
        norm="gn",
        stage_with_dcn=[False, True, True, True],
        dcn_config=dict(
            modulated=True,
            deformable_groups=1
        )
)
neck=dict(
        type="SegDBNeck",
        in_channels=[64,128,256,512],
        inner_channels = 256,
)
det_head=dict(
        type="DBHead",
        inner_channels=256,
        neck_out_channels=256 // 4,
        k=5,
        max_candidates=1000,
)
backbone_model = build_backbone(backbone).cuda()
neck_model = build_neck(neck).cuda()
head_model = build_head(det_head).cuda()

from texthub.utils import Config
from texthub.datasets import build_dataset

config_file = "./configs/detection/DB/resnet18_deform.py"
cfg = Config.fromfile(config_file)
##
dataset = build_dataset(cfg.data.train)
import torch
b=torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            num_workers=4,
            shuffle=False,
            pin_memory=True
        )
data=b.__iter__().__next__()

# def batch_dict_data_tocuda(data:dict):
#     for key,values in data.items():
#         if hasattr(values,'cuda'):
#             data[key]=values.cuda()
#     return data
#
# data = batch_dict_data_tocuda(data)
# img_tensor = backbone_model(data['img'])
# img_tensor = neck_model(img_tensor)
# data["img"] = img_tensor
# result = head_model(data,return_loss=True)
# print(result)
def toPILImage(img_tensor:torch.Tensor)->Image:
    image = transforms.ToPILImage()(img_tensor)
    return image

toPILImage(data["img"][0]).show()
toPILImage(data["gt"][0]).show("gt")
toPILImage(data["mask"][0]).show("mask")
toPILImage(data["thresh_map"][0]).show("thresh_map")
toPILImage(data["thresh_mask"][0]).show("trainning_mask")