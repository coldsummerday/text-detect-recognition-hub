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
# config_file = "./configs/recognition/fourstagerecogition/tps_resnet_lstm_attention_chi_iter_maskresize.py"
config_file = "../configs/detection/DB/db_resnet18_deform.py"
cfg = Config.fromfile(config_file)
##


def toPILImage(img_tensor:torch.Tensor)->Image:
    image = transforms.ToPILImage()(img_tensor)
    return image


dataset = build_dataset(cfg.data.train)
import torch
b=torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=1,
    pin_memory=True,
    shuffle=True,
        )
def datasetitemn2show(data:dict):
    for key,value in data.items():
        if type(value[0]) != str:
            print(key, value[0], value[0].shape)
        else:
            print(key, value[0])
        if isinstance(value,np.ndarray) or isinstance(value,torch.Tensor):
            img =toPILImage(value[0])
            img.save('./testimgs/{}.jpg'.format(key))
        print(key,value)





import matplotlib.pyplot as plt
def show_img(imgs: np.ndarray, color=False):
    if (len(imgs.shape) == 3 and color) or (len(imgs.shape) == 2 and not color):
        imgs = np.expand_dims(imgs, axis=0)
    for img in imgs:
        plt.figure()
        plt.imshow(img, cmap=None if color else 'gray')
#
# for i,data in enumerate(b):
#     img = data["img"]
#     label = data["gt"]
#     mask = data["mask"]
#     print(label.shape)
#     print(img.shape)
#     print(label[0][-1].sum())
#     print(mask[0].shape)
#     # pbar.update(1)
#     show_img((img[0] * mask[0].to(torch.float)).numpy().transpose(1, 2, 0), color=True)
#     show_img(label[0])
#     show_img(mask[0])
#     plt.show()
#
data=b.__iter__().__next__()
# datasetitemn2show(data)

gt_numpy = data["gt"].cpu().numpy()
thresh_map_numpy =data["thresh_map"].cpu().numpy()
index = 0

# index = 0
# from tqdm import tqdm
# for data in tqdm(b):
#     img_tensors = data["img"]
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

