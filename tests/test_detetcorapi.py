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
import matplotlib.pyplot as plt


set_random_seed(12)
config_file = "./configs/detection/PSE/pse_resnet50_deform_6_trainner.py"

checkpoint = "./work_dirs/pse_resnet50_deform_6_epoch_trainer/PSEDetector_epoch_15.pth"
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

def show_img(imgs: np.ndarray, color=False):
    if (len(imgs.shape) == 3 and color) or (len(imgs.shape) == 2 and not color):
        imgs = np.expand_dims(imgs, axis=0)
    for img in imgs:
        plt.figure()
        plt.imshow(img, cmap=None if color else 'gray')



# from tqdm import tqdm
# # img = "./testimgs/img_305.jpg"
# eval_path = "/home/zhou/data/data/receipt/end2end/receipt_2nd_icdr15val/imgs/"
# img_paths = os.listdir(eval_path)
# for img_id in tqdm(img_paths):
#     img_path = os.path.join(eval_path,img_id)
#
#     preds,scores = inference_detector(model,img_path)
#     img = cv2.imread(img_path)
#     img = draw_bbox(img, preds)
#     cv2.imwrite("./testimgs/{}".format(img_id),img)
img = "/home/zhou/data/data/receipt/end2end/receipt_2nd_icdr15/ori_imgs/12.jpg"
preds,scores  = inference_detector(model,img)
# img = cv2.imread(img)
# print(preds)
# img = draw_bbox(img,preds)
plt.show()

# cv2.waitKey()
# show_img(img)