import os.path as osp
import os
import sys
this_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(osp.join(this_path,'../'))
from texthub.apis import init_recognizer,inference_recognizer
import torch
config_file ="./configs/receipt/recognition/plugnet.py"
checkpoint = "./work_dirs/plugnet/plugnet_lr/PlugNet_epoch_60.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_recognizer(config_file,checkpoint,device)

img = "./testimgs/hr_img.jpg"
print(inference_recognizer(model,img))