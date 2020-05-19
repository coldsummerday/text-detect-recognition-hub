from texthub.utils import Config
from texthub.datasets import build_dataset
import numpy as np
import cv2
config_file = "./configs/testpandetect.py"
cfg = Config.fromfile(config_file)
##
dataset = build_dataset(cfg.data.train)
b=dataset[0]

def show_pic(img, bboxes=None, name='pic'):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    show_img = img.copy()
    if not isinstance(bboxes, np.ndarray):
        bboxes = np.array(bboxes)
    for point in bboxes.astype(np.int):
        cv2.line(show_img, tuple(point[0]), tuple(point[1]), (255, 0, 0), 2)
        cv2.line(show_img, tuple(point[1]), tuple(point[2]), (255, 0, 0), 2)
        cv2.line(show_img, tuple(point[2]), tuple(point[3]), (255, 0, 0), 2)
        cv2.line(show_img, tuple(point[3]), tuple(point[0]), (255, 0, 0), 2)
    # cv2.namedWindow(name, 0)  # 1表示原图
    # cv2.moveWindow(name, 0, 0)
    # cv2.resizeWindow(name, 1200, 800)  # 可视化的图片大小
    cv2.imshow(name, show_img)
    cv2.waitKey()

import torch
b=torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            pin_memory=True
        )
b.__iter__().__next__()