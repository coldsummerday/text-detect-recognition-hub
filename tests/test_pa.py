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


def draw_bbox(img_path, result, color=(0, 255, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, color)

        # cv2.line(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
        # cv2.line(img_path, tuple(point[1]), tuple(point[2]), color, thickness)
        # cv2.line(img_path, tuple(point[2]), tuple(point[3]), color, thickness)
        # cv2.line(img_path, tuple(point[3]), tuple(point[0]), color, thickness)
    return img_path


set_random_seed(12)
# config_file = "../configs/detection/DB/db_resnet18_deform_border.py"
#
# checkpoint = "../work_dirs/db/db_resnet18_deform_adam/DBDetector_epoch_120.pth"
config_file = "./configs/detection/pan/pan_new_pa.py"
checkpoint = "./work_dirs/pan_cpp/PAN_epoch_5.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_detector(config_file,checkpoint,device)


# img = "/home/zhou/data/data/receipt/end2end/receipt_2nd_icdr15/ori_imgs/12.jpg"
# img = "/home/zhou/data/data/receipt/end2end/receipt_2nd_icdr15val/imgs/img_9957.jpg"
# preds,scores  = inference_detector(model,img)
# img = cv2.imread(img)
# img = draw_bbox(img,preds)
# cv2.imwrite("./testimgs/{}".format("9957.jpg"),img)

eval_path = "/data/zhb/data/receipt/end2end/receipt_2nd_icdr15val/imgs/"
img_paths = os.listdir(eval_path)
from tqdm import tqdm
for img_id in  tqdm(img_paths):
    img_path = os.path.join(eval_path,img_id)
    preds,scores = inference_detector(model,img_path)
    img = cv2.imread(img_path)
    img = draw_bbox(img, preds)
    cv2.imwrite("./testimgs/{}".format(img_id),img)
    #img_9957.jpg