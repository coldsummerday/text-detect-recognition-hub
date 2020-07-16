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
config_file = "./configs/testpandetect.py"
checkpoint = "./work_dirs/pan/PAN_epoch_24.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_detector(config_file,checkpoint,device)

def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.line(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
        cv2.line(img_path, tuple(point[1]), tuple(point[2]), color, thickness)
        cv2.line(img_path, tuple(point[2]), tuple(point[3]), color, thickness)
        cv2.line(img_path, tuple(point[3]), tuple(point[0]), color, thickness)
    return img_path



# img = "./testimgs/img_305.jpg"
img = "/home/zhou/data/data/receipt/end2end/receipt_2nd_icdr15/ori_imgs/8.jpg"
preds = inference_detector(model,img)
img = cv2.imread(img)

print(len(preds))

# img = draw_bbox(img,preds)
# cv2.imshow("s",img)
# cv2.imwrite("./testimgs/1.jpeg",img)
# cv2.imshow("s",img)
# cv2.waitKey()

