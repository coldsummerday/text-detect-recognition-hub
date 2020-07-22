import os.path as osp
import os
import sys
import cv2
import  numpy as  np
this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(osp.join(this_path,'../'))
from texthub.utils import Config
from texthub.datasets import build_dataset
import  torch
from PIL import Image
from torchvision import transforms
config_file = "./configs/recognition/fourstagerecogition/tps_resnet_lstm_attention_chi_iter_maskresize.py"
cfg = Config.fromfile(config_file)
##


def toPILImage(img_tensor:torch.Tensor)->Image:
    image = transforms.ToPILImage()(img_tensor)
    return image


dataset = build_dataset(cfg.data.train)
import torch
b=torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            num_workers=4,
            shuffle=False,
            pin_memory=True
        )
index = 0
from tqdm import tqdm
for data in tqdm(b):
    img_tensors = data["img"]
    #尝试是否能读取完
#     for i in img_tensors:
#         img = toPILImage(i)
#         img.save('./testimgs/mask_{}.jpg'.format(index))
#         index += 1
#     if index==20:
#         break
# data=b.__iter__().__next__()

# def main():
#     config_file = "../configs/testdatasetconfig.py"
#     cfg = Config.fromfile(config_file)
#     dataset = build_dataset(cfg.data.train)

