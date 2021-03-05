import  PIL
from PIL import ImageFont,ImageDraw,Image
import  numpy as np
import  cv2
import os
this_path = os.path.split(os.path.realpath(__file__))[0]

class GenRecognitionImageCV2(object):
    """
    文本识别的resize
    img_scale:(h,w)
    尽量不resize 图片，只将原始图片贴到mask上，保持图像原来清晰度
    """
    def __init__(self,img_scale=None,img_channel = 1,font_file="sont.ttf",font_color="black"):
        assert  isinstance(img_scale,tuple) and len(img_scale)==2,"img_scale must be tuple(h,w)"
        self.default_h,self.default_w = img_scale
        self.img_channel = img_channel
        self.font_handler = ImageFont.truetype(os.path.join(this_path,'../resources/',font_file),int(self.default_h/4*3),encoding="utf-8")
        # self.font_handler = ImageFont.truetype("/home/zhou/project/receipt/text-detect-recognition-hub/texthub/datasets/resources/sont.ttf",int(self.default_h/4*3),encoding="utf-8")
        self.font_color_value = 0
        if font_color=="white":
            self.font_color_value = 255


    def __call__(self, data:{}):

        # img
        img = data["img"]
        text = data["text"]
        if img.ndim==2:
            ori_h,ori_w = img.shape
            border_pixel = img[ori_h - 1, ori_w - 1]
            mask = np.full((self.default_h, self.default_w), border_pixel).astype("float32")
            pilimg = Image.fromarray(mask)
            draw = ImageDraw.Draw(pilimg)
            draw.text((0, 0), text, self.font_color_value, font=self.font_handler)
            gen_img = np.array(pilimg)
            data["gen_img"]=gen_img
            return data
        else:
            ori_h,ori_w,_ = img.shape
            # cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = Image.new("RGBA",(self.default_w,self.default_h),(int(img[ori_h - 1, ori_w - 1][2]),int(img[ori_h - 1, ori_w - 1][1]),int(img[ori_h - 1, ori_w - 1][0]),0))
            # border_pixel = img[ori_h - 1, ori_w - 1]
            # mask = np.zeros((self.default_h, self.default_w,3)).astype("float32")
            # mask[:,:,0]= np.full((self.default_h, self.default_w), border_pixel[0]).astype("float32")
            # mask[:, :, 1] = np.full((self.default_h, self.default_w), border_pixel[1]).astype("float32")
            # mask[:, :, 2] = np.full((self.default_h, self.default_w), border_pixel[2]).astype("float32")
            # cv2img = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            #
            # pilimg = Image.fromarray(cv2img)
            draw = ImageDraw.Draw(mask)  # 图片上打印
            draw.text((0, 0), text, (self.font_color_value, self.font_color_value, self.font_color_value), font=self.font_handler)
            gen_img = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
            data["gen_img"] = gen_img
            return data
        return data

text = "我是谁"
img = np.ones((32,100))
data = {"img":img,"text":text}
gen = GenRecognitionImageCV2((32,100),1)
new_data = gen(data)
cv2.imwrite("new.jpg",new_data["gen_img"])



import torch
from torch import nn
from texthub.modules.backbones.botnet import BottleStack

layer = BottleStack(
    dim = 256,              # channels in
    fmap_size = 64,         # feature map size
    dim_out = 2048,         # channels out
    proj_factor = 4,        # projection factor
    downsample = True,      # downsample on first layer or not
    heads = 4,              # number of heads
    dim_head = 128,         # dimension per head, defaults to 128
    rel_pos_emb = False,    # use relative positional embedding - uses absolute if False
    activation = nn.ReLU()  # activation throughout the network
)

fmap = torch.randn(2, 256, 64, 64) # feature map from previous resnet block(s)

import  torch.nn.functional as F
layer(fmap) # (2, 2048, 32, 32)
import torch
img_h,img_w = 48,160
from texthub.modules.backbones.rec_botresnet import AsterBotResNet
model = AsterBotResNet(input_channel=1,input_size=(img_h,img_w),output_channel=512)
tensor = torch.randn((3,1,img_h,img_w))
print(model(tensor).shape)

import torch.nn as nn

class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.a = nn.Linear(2,3)
        self.b = nn.Linear(3,4)
        self.c = nn.Linear(2,3)


print(A)

import torch.nn as nn
nn.GRU
nn.GRUCell




