import os.path as osp
import os
import sys
import cv2
import  numpy as  np
this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(osp.join(this_path,'../'))
from texthub.modules.builder import  build_backbone,build_neck,build_recognizer
from texthub.utils import Config
from texthub.datasets import build_dataset

config_file = "./configs/receipt/recognition/2dctc/2dctc_charset.py"
cfg = Config.fromfile(config_file)

model = build_recognizer(cfg.model)

##
# dataset = build_dataset(cfg.data.train)
# import torch
# b=torch.utils.data.DataLoader(
#             dataset,
#             batch_size=64,
#             pin_memory=True,
# drop_last=True
# )
#
# show_index = 200
# backbone = build_backbone(cfg.model.backbone).cuda()
# nexk = build_neck(cfg.model.neck).cuda()
# for index,data_dict in enumerate(b):
#     x = backbone(data_dict['img'].cuda())
#     x = nexk(x)
#     if index % show_index==0:
#         print(index)
#

















