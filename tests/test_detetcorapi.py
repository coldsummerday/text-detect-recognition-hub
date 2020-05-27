import os.path as osp
import os
import sys
import cv2
import numpy as np
this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(osp.join(this_path,'../'))
from texthub.apis import init_detector,inference_detector
import torch
config_file = "./configs/testpandetect.py"
checkpoint = "./work_dirs/pan/PAN_epoch_24.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_detector(config_file,checkpoint,device)



img = "./testimgs/img_306.jpg"
plogs = inference_detector(model,img)
img = cv2.imread(img)
img=cv2.resize(img,(640,640))
for j in plogs:
    for i in j:
        xmin,xmax,ymin,ymax = i.boundingBox()
        xmin,xmax,ymin,ymax = int(xmin),int(xmax),int(ymin),int(ymax)
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),2)

cv2.imshow("s",img)
cv2.waitKey()
