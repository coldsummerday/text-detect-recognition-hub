import os.path as osp
import os
import sys
import cv2
import  numpy as  np
this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(osp.join(this_path,'../'))
from texthub.apis import init_detector,inference_detector
from texthub.utils import set_random_seed
import torch

set_random_seed(12)
config_file = "./configs/detection/DB/db_resnet18_defrom_epoch.py"
checkpoint = "./work_dirs/db_resnet18_deform_epoch/epoch_1200.pth"
# config_file = "./configs/detection/pan/pandetect.py"
# checkpoint = "./work_dirs/pan/PAN_epoch_22.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_detector(config_file,checkpoint,device)

def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, (0, 255, 255))

        # cv2.line(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
        # cv2.line(img_path, tuple(point[1]), tuple(point[2]), color, thickness)
        # cv2.line(img_path, tuple(point[2]), tuple(point[3]), color, thickness)
        # cv2.line(img_path, tuple(point[3]), tuple(point[0]), color, thickness)
    return img_path
#
#
from tqdm import tqdm
img = "/home/zhou/data/data/receipt/end2end/receipt_2nd_icdr15val/imgs/img_9806.jpg"
preds,scores = inference_detector(model,img)
#
# eval_path = "/home/zhou/data/data/receipt/end2end/receipt_2nd_icdr15val/imgs/"
# img_paths = os.listdir(eval_path)
# for img_id in tqdm(img_paths):
#     img_path = os.path.join(eval_path,img_id)
#
#     preds,scores = inference_detector(model,img_path)
#     img = cv2.imread(img_path)
#     img = draw_bbox(img, preds)
#     cv2.imwrite("./testimgs/{}".format(img_id),img)
#
